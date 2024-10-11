import time
import torch
import numpy as np
import ctypes
import nvtx 

import BAM_Feature_Store
from BAM_Feature_Store import Emulate_SA

from Dist_Cache import NVSHMEM_Cache

import dgl
from torch.utils.data import DataLoader
from collections.abc import Mapping

from dgl.dataloading import create_tensorized_dataset, WorkerInitWrapper, remove_parent_storage_columns
from dgl.utils import (
    recursive_apply, ExceptionWrapper, recursive_apply_pair, set_num_threads, get_num_threads,
    get_numa_nodes_cores, context_of, dtype_of)

from dgl import DGLHeteroGraph
from dgl.frame import LazyFeature
from dgl.storages import wrap_storage
from dgl.dataloading.base import BlockSampler, as_edge_prediction_sampler
from dgl import backend as F
from dgl.distributed import DistGraph
from dgl.multiprocessing import call_once_and_share

import torch.distributed as dist


from collections import Counter


def _get_device(device):
    device = torch.device(device)
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    return device

class CollateWrapper(object):
    def __init__(self, sample_func, g,  device):
        self.sample_func = sample_func
        self.g = g
        self.device = device
        self.pin_memory = True



    def __call__(self, items):
        graph_device = getattr(self.g, 'device', None)   
        items = recursive_apply(items, lambda x: x.to(self.device))
        with nvtx.annotate("Sample", color="green"):
            batch = self.sample_func(self.g, items)

        return batch


class GIDS_NVShmem_Loader_PrefetchingIter(object):
    def __init__(self, dataloader, dataloader_it, GIDS_Loader=None):
        self.dataloader_it = dataloader_it
        self.dataloader = dataloader
        self.graph_sampler = self.dataloader.graph_sampler
        self.GIDS_Loader=GIDS_Loader

    def __iter__(self):
        return self

    def __next__(self):
        cur_it = self.dataloader_it
        batch = self.GIDS_Loader.fetch_feature(self.dataloader.dim, cur_it, self.GIDS_Loader.gids_device)
        return batch



class GIDS_NVShmem_Loader(torch.utils.data.DataLoader):

    def __init__(self, graph, indices, graph_sampler, batch_size, dim, GIDS, device=None, use_ddp=False,
                 ddp_seed=0, drop_last=False, shuffle=False,
                 use_alternate_streams=None, 
                 
                 nvshmem_batch = None,
                 **kwargs):

        self.nvshmem_batch = None
        self.use_nvshmem_batch = False
        if(nvshmem_batch is not None):
            self.nvshmem_batch = GIDS.nvshmem_malloc()
            self.use_nvshmem_batch = True


        use_uva = False
        self.GIDS_Loader = GIDS
        self.dim = dim

        if isinstance(kwargs.get('collate_fn', None), CollateWrapper):
            assert batch_size is None       # must be None
            # restore attributes
            self.graph = graph
            self.indices = indices
            self.graph_sampler = graph_sampler
            self.device = device
            self.use_ddp = use_ddp
            self.ddp_seed = ddp_seed
            self.shuffle = shuffle
            self.drop_last = drop_last
            self.use_alternate_streams = use_alternate_streams
            self.use_uva = use_uva
            kwargs['batch_size'] = None
            super().__init__(**kwargs)
            return


        if isinstance(graph, DistGraph):
            raise TypeError(
                'Please use dgl.dataloading.DistNodeDataLoader or '
                'dgl.datalaoding.DistEdgeDataLoader for DistGraphs.')
  
        self.graph = graph
        self.indices = indices     
        num_workers = kwargs.get('num_workers', 0)


        indices_device = None
        try:
            if isinstance(indices, Mapping):
                indices = {k: (torch.tensor(v) if not torch.is_tensor(v) else v)
                           for k, v in indices.items()}
                indices_device = next(iter(indices.values())).device
            else:
                indices = torch.tensor(indices) if not torch.is_tensor(indices) else indices
                indices_device = indices.device
        except:     # pylint: disable=bare-except
            # ignore when it fails to convert to torch Tensors.
            pass

        if indices_device is None:
            if not hasattr(indices, 'device'):
                raise AttributeError('Custom indices dataset requires a \"device\" \
                attribute indicating where the indices is.')
            indices_device = indices.device

        if device is None:     
            device = torch.cuda.current_device()
        self.device = _get_device(device)
        
        #pin graph
        if not self.graph._graph.is_pinned():
            print("\t\tPinning graph")
            self.graph._graph.pin_memory_()
            
        else:
            print('\t\tNot pinning Graph')


        # Sanity check - we only check for DGLGraphs.
        if isinstance(self.graph, DGLHeteroGraph):            
            self.graph.create_formats_()
            if not self.graph._graph.is_pinned():
                self.graph._graph.pin_memory_()
            

            # Check use_alternate_streams
            if use_alternate_streams is None:
                use_alternate_streams = (
                    self.device.type == 'cuda' and self.graph.device.type == 'cpu' and
                    not use_uva)

        if (torch.is_tensor(indices) or (
                isinstance(indices, Mapping) and
                all(torch.is_tensor(v) for v in indices.values()))):
            self.dataset = create_tensorized_dataset(
                indices, batch_size, drop_last, use_ddp, ddp_seed, shuffle,
                kwargs.get('persistent_workers', False))
        else:
            self.dataset = indices

        self.ddp_seed = ddp_seed
        self.use_ddp = use_ddp
        self.use_uva = use_uva
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.graph_sampler = graph_sampler
        self.use_alternate_streams = use_alternate_streams


        self.cpu_affinity_enabled = False

        worker_init_fn = WorkerInitWrapper(kwargs.get('worker_init_fn', None))

        self.other_storages = {}

        super().__init__(
            self.dataset,
            collate_fn=CollateWrapper(
                self.graph_sampler.sample, graph, self.device),
            batch_size=None,
            pin_memory=False,
            worker_init_fn=worker_init_fn,
            **kwargs)

    def __iter__(self):
        if self.shuffle:
            self.dataset.shuffle()

        return GIDS_NVShmem_Loader_PrefetchingIter(
            self, super().__iter__(), GIDS_Loader=self.GIDS_Loader)
 
    def print_stats(self):
        self.GIDS_Loader.print_stats()

    def print_timer(self):
        self.sample_time = 0.0
        self.graph_travel_time = 0.0





class NVSHMEM_GIDS():
    def __init__(self, page_size=4096, off=0, dim = 1024, cache_dim = 1024, num_ele = 300*1000*1000*1024, 
        num_ssd = 1,  
        cache_size = 10,  num_ways=4,
        ctrl_idx=0, 
        heterograph=False,
        heterograph_map=None,
        use_ddp = False, 
        fan_out = None, 
        batch_size = 1024,
        heterogeneous = False, 
        hetero_map=None):

 
        #DDP parameters
        self.use_ddp = use_ddp

        self.NVSHMEM_Cache = Dist_Cache.NVSHMEM_Cache()
        # self.NVSHMEM_Cache.init_mpi_nvshmem()
        # self.rank = self.NVSHMEM_Cache.get_rank()
        # self.world_size = self.NVSHMEM_Cache.get_world_size()
        # print(f"Hello from rank {self.rank} out of {self.world_size} processes")

        self.rank = 0
        #nvshmem_ptr = self.NVSHMEM_Cache.nvshmem_malloc(128)
        #print(f"Rank:{self.rank} nvshmem ptr: {nvshmem_ptr}")

        #FIX, jut for test
        self.world_size = 1

        self.heterogeneous = heterogeneous
        self.hetero_map = hetero_map

        if(self.heterogeneous):
            if(hetero_map == None):
                print("Need key-offset map for heterogeneous graph")  

        self.max_sample_size = batch_size
        self.fan_out = fan_out
        
        for i in fan_out:
            self.max_sample_size *= i
        print("max sample size: ", self.max_sample_size)

 


        # Cache Parameters
        self.page_size = page_size
        self.off = off
        self.num_ele = num_ele
        self.cache_size = cache_size
        self.num_ways = num_ways
        self.num_ssd = num_ssd


      

        #True if the graph is heterogenous graph
        self.heterograph = heterograph
        self.heterograph_map = heterograph_map
        self.graph_GIDS = None

        #FIX
        self.cache_dim = dim
        #self.gids_device="cuda:" + str(ctrl_idx)

        self.GIDS_controller = BAM_Feature_Store.GIDS_Controllers()



    
        #TIMERS
        self.GIDS_time = 0.0
        self.WB_time = 0.0
        
        self.Sampling_time = 0.0

        self.Split_time = 0.0
        self.Communication_time = 0.0
        self.Communication_setup_time = 0.0
        self.Gather_time = 0.0
        self.agg_time = 0.0
        self.Reset_time = 0.0


    def get_rank(self):
        return self.rank
    def get_world_size(self):
        return self.world_size 

    def init_cache(self, device_id, ssd_list = None):

        self.gids_device="cuda:" + str(device_id)
        self.device_id = device_id

        if (ssd_list == None):
            print("SSD are not assigned")
            self.ssd_list = [i for i in range(self.num_ssd)] 
        else:
            self.ssd_list = ssd_list

        print("SSD list: ", self.ssd_list, " Device id: ", device_id)
        self.GIDS_controller.init_GIDS_controllers(self.num_ssd, 1024, 128, self.ssd_list, device_id)
        self.NVSHMEM_Cache.init_cache(self.GIDS_controller, self.page_size, self.off, self.cache_size, self.world_size, self.num_ele, self.num_ssd, self.num_ways)

    
    
    #Fetching Data from the SSDs
    def fetch_feature(self, dim, it, device):
        batch =  self.SA_fetch_feature(dim, it, device)
        return batch


    #Fetching Data from the SSDs
    def SA_fetch_feature(self, dim, it, device):
        GIDS_time_start = time.time()

        

        batch = next(it)

  
     
        if(self.heterograph):
            print("HETERO GRAPH IS NOT SUPPORT YET")
            # ret_ten = {}
            # index_size_list = []
            # index_ptr_list = []
            # return_torch_list = []
            # key_list = []
            
            # num_keys = 0
            # for key , v in batch[0].items():
            #     if(len(v) == 0):
            #         empty_t = torch.empty((0, dim)).to(self.gids_device).contiguous()
            #         ret_ten[key] = empty_t
            #     else:
            #         key_off = 0
            #         if(self.heterograph_map != None):
            #             if (key in self.heterograph_map):
            #                 key_off = self.heterograph_map[key]
            #             else:
            #                 print("Cannot find key: ", key, " in the heterograph map!")
                    
            #         g_index = v.to(self.gids_device)
            #         index_size = len(g_index)
            #         index_ptr = g_index.data_ptr()
                    
            #         return_torch =  torch.zeros([index_size,dim], dtype=torch.float, device=self.gids_device).contiguous()
            #         return_torch_list.append(return_torch.data_ptr())
            #         ret_ten[key] = return_torch
            #         num_keys += 1
            #         index_ptr_list.append(index_ptr)
            #         index_size_list.append(index_size)
            #         key_list.append(key_off)

            # self.BAM_FS.read_feature_hetero(num_keys, return_torch_list, index_ptr_list, index_size_list, dim, self.cache_dim, key_list)

            # batch.append(ret_ten)
            # self.GIDS_time += time.time() - GIDS_time_start
            return batch

        else:
            index = batch[0].to(self.gids_device)
            index_size = len(index)
            index_ptr = index.data_ptr()
            return_torch =  torch.zeros([index_size,dim], dtype=torch.float, device=self.gids_device).contiguous()

            #print("index: ", index)

            self.NVSHMEM_Cache.read_feature(return_torch.data_ptr(), index_ptr, index_size, dim, self.cache_dim)
            self.GIDS_time += time.time() - GIDS_time_start
           # print("return torch: ", return_torch)
            batch = (*batch, return_torch)
            return batch

    def print_stats(self):
        self.NVSHMEM_Cache.print_stats()

    def nvshmem_malloc(self):
        return self.NVSHMEM_Cache.nvshmem_malloc(100 * 4)

    def __del__(self):
        """
        Destructor to clean up and finalize MPI.
        """
        # Check if NVSHMEM_Cache has been initialized to avoid errors
        if hasattr(self, 'NVSHMEM_Cache'):
            try:
                # Call finalize_mpi if NVSHMEM_Cache is initialized
                self.NVSHMEM_Cache.finalize_mpi()
                print(f"MPI finalized successfully for rank {self.rank}.")
            except Exception as e:
                print(f"Error during MPI finalization for rank {self.rank}: {e}")

    

    