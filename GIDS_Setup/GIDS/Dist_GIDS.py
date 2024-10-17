import time
import torch
import numpy as np
import ctypes
import nvtx 

import BAM_Feature_Store
#from BAM_Feature_Store import Emulate_SA


import Dist_Cache
from Dist_Cache import NVSHMEM_Cache, Dist_GIDS_Controllers

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

import nvshmem_manager
import cupy as cp


class  Emulate_Cache_CollateWrapper(object):
    def __init__(self, sample_func, g,  device, emul_SA, emul_SA_list, dim, cache_dim, ep, cache_track, distribute_method):
        self.sample_func = sample_func
        self.g = g
        self.device = device
        self.pin_memory = True

        self.Emul_SA = emul_SA
        self.Emul_SA_list = emul_SA_list
        self.cache_track = cache_track

        self.distribute_method = distribute_method
        self.dim = dim
        self.cache_dim = cache_dim
        self.eviction_policy = ep

    def __call__(self, items):
        item_list = self.distribute_node(items)
        init_batch = None
        for i in range(self.cache_track.num_gpus):
            batch = self.sample_func(self.g, item_list[i])
            if(i == 0):
                init_batch = batch
            index_size = len(batch[0])
            index = batch[0].to(self.device)
            return_torch =  torch.zeros([index_size, self.dim], dtype=torch.float, device=self.device)
            color_tensor = torch.zeros([index_size], dtype=torch.int64, device=self.device)

            index_cpu = index.cpu()
            color_tensor_cpu = self.cache_track.color_tensor[index_cpu]
            color_tensor = color_tensor_cpu.to('cuda:0')

            static_info_ten = None
            if(self.eviction_policy == 1):
                static_info_ten = self.get_static_info(index)
                self.Emul_SA_list[i].read_feature_with_color(return_torch.data_ptr(), index.data_ptr(), index_size, self.dim, self.cache_dim, 0, static_info_ten.data_ptr(), color_tensor.data_ptr())                 

            else:
                self.Emul_SA_list[i].read_feature_with_color(return_torch.data_ptr(), index.data_ptr(), index_size, self.dim, self.cache_dim, 0, 0, color_tensor.data_ptr())                 

        return init_batch


class NVShmem_Tensor_Manager(object):
    def __init__(self, nvshmem_batch_ptr: int, batch_nbytes: int, nvshmem_index_ptr: int, index_nbytes: int, shape: tuple, device: str):
        self.nvshmem_batch_ptr = nvshmem_batch_ptr
        self.batch_nbytes = batch_nbytes
        self.batch_mem = cp.cuda.UnownedMemory(nvshmem_batch_ptr, batch_nbytes, owner=None)
        self.batch_memptr = cp.cuda.MemoryPointer(self.batch_mem, offset=0)

        self.nvshmem_index_ptr = nvshmem_index_ptr
        self.index_nbytes = index_nbytes
        
        self.index_mem = cp.cuda.UnownedMemory(nvshmem_index_ptr, index_nbytes, owner=None)
        self.index_memptr = cp.cuda.MemoryPointer(self.index_mem, offset=0)


        self.device = device

        # Shape of Index Tensor does not change, so generate only once
        arr = cp.ndarray(shape, dtype=cp.int64, memptr=self.index_memptr)
        self.index_tensor =  torch.as_tensor(arr, device=self.device)

    def get_batch_tensor(self, shape: tuple):
        arr = cp.ndarray(shape, dtype=cp.float32, memptr=self.batch_memptr)
        return torch.as_tensor(arr, device=self.device)

    def get_batch_tensor_ptr(self):
        return self.nvshmem_batch_ptr

    def get_index_tensor(self):
        return self.index_tensor

    def get_index_tensor_ptr(self):
        return self.nvshmem_index_ptr



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
        item_test = items.to(self.device)

        items = recursive_apply(items, lambda x: x.to(self.device))
        batch = self.sample_func(self.g, items)
        batch = recursive_apply(batch, remove_parent_storage_columns, self.g)
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
        # if(nvshmem_batch is not None):
        #     self.nvshmem_batch = GIDS.nvshmem_malloc()
        #     self.use_nvshmem_batch = True


        use_uva = True
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
            print("setting device: ", device)
        self.device = _get_device(device)
        
        # self.device = "cpu"
        print("Current setting device: ", self.device)

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
        GPU_cache_size = 10,  
        CPU_cache_size = 0, 
        num_ways=32, 
        ctrl_idx=0, 
        heterograph=False,
        heterograph_map=None,
        use_ddp = False, 
        fan_out = None, 
        batch_size = 1024,
        heterogeneous = False, 
        hetero_map=None,
        use_nvshmem_tensor = False,
        nvshmem_test = False,
        is_simulation = False,
        feat_file = "",
        feat_off = 0):

 
        #DDP parameters
        self.use_ddp = use_ddp
        self.use_nvshmem_tensor = use_nvshmem_tensor


        self.is_simulation = is_simulation
        self.feat_file = feat_file
        self.feat_off = feat_off
        # Flag for NVshme Tensor Micro benchmark
        self.nvshmem_test = nvshmem_test

        self.NVSHMEM_Cache = Dist_Cache.NVSHMEM_Cache()
        

        self.NVshmem_tensor_manager = None
        self.batch_size = batch_size
        self.max_sample_size = batch_size
        self.fan_out = fan_out

        self.cache_dim = dim
        self.dim = dim
        
        for i in fan_out:
            self.max_sample_size *= i
        self.max_sample_size *= 2
        print("max sample size: ", self.max_sample_size)

        self.cache_per_GPU = 1
        self.NVSHMEM_Cache.init()
        self.rank = self.NVSHMEM_Cache.get_rank()
        self.world_size = self.NVSHMEM_Cache.get_world_size()
        self.mype_node = self.NVSHMEM_Cache.get_mype_node()

        if(self.use_nvshmem_tensor or self.nvshmem_test):
            #self.NVshmem_manager =  nvshmem_manager.NVSHMEMWrapper()
       

            # self.NVSHMEM_Cache.init()
            # self.rank = self.NVSHMEM_Cache.get_rank()
            # self.world_size = self.NVSHMEM_Cache.get_world_size()
            # self.mype_node = self.NVSHMEM_Cache.get_mype_node()
            self.cache_per_GPU = self.world_size
            
            nbytes = int(self.max_sample_size * 4 * dim)
            nvshmem_ptr = self.NVSHMEM_Cache.allocate(nbytes) 
            print(f"Rank:{self.rank} nvshmem ptr: {nvshmem_ptr}")

            # Number of GPUs * Batch Size * sizeof(int64_t) * pair
            index_nbytes = int(self.max_sample_size  * 8 * 2)
            nvshmem_index_ptr = self.NVSHMEM_Cache.allocate(nbytes) 
            print(f"Rank:{self.rank} nvshmem ptr: {nvshmem_ptr}")

            index_shape = [self.world_size, int(self.max_sample_size * 2)]

            device = "cuda:" + str(self.rank)
            self.NVshmem_tensor_manager = NVShmem_Tensor_Manager(nvshmem_ptr, nbytes, nvshmem_index_ptr, index_nbytes, index_shape, device)
            
            self.NVSHMEM_Cache.update_NVshmem_metadata(self.rank, self.world_size, self.mype_node)

            #self.world_size = 1
            # Micro Benchmark, independent cache with NVShmem memory
            if(self.nvshmem_test):
                #self.world_size = 1
                self.cache_per_GPU = 1

        else:
            #self.NVshmem_manager =  nvshmem_manager.NVSHMEMWrapper()
            #self.NVSHMEM_Cache.init()
            #self.rank = self.NVSHMEM_Cache.get_rank()
            #self.rank = 0
            self.cache_per_GPU = 1
            #self.world_size = 1


        #FIX, jut for test

        self.heterogeneous = heterogeneous
        self.hetero_map = hetero_map

        if(self.heterogeneous):
            if(hetero_map == None):
                print("Need key-offset map for heterogeneous graph")  

  
 


        # Cache Parameters
        self.page_size = page_size
        self.off = off
        self.num_ele = num_ele
        self.GPU_cache_size = GPU_cache_size
        self.CPU_cache_size = CPU_cache_size
        self.num_ways = num_ways
        self.num_ssd = num_ssd


      

        #True if the graph is heterogenous graph
        self.heterograph = heterograph
        self.heterograph_map = heterograph_map
        self.graph_GIDS = None

        #FIX
        self.gids_device="cuda:" + str(self.rank)

        #self.GIDS_controller = BAM_Feature_Store.GIDS_Controllers()
        self.GIDS_controller =  Dist_GIDS_Controllers()



    
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
        self.GIDS_controller.init_GIDS_controllers(self.num_ssd, 1024, 128, self.ssd_list, device_id, self.is_simulation)
        self.NVSHMEM_Cache.init_cache(self.GIDS_controller, self.page_size, self.off, self.GPU_cache_size, self.CPU_cache_size, self.cache_per_GPU, self.num_ele, self.num_ssd, self.num_ways, self.is_simulation, self.feat_file, self.feat_off)
    
    
    #Fetching Data from the SSDs
    def fetch_feature(self, dim, it, device):
        if(self.use_nvshmem_tensor):
            batch = self.Dist_Cache_fetch_feature(dim, it, device)
        else:
            batch =  self.Independent_Cache_fetch_feature(dim, it, device)
        return batch


    def Dist_Cache_fetch_feature(self, dim, it, device):
        GIDS_time_start = time.time()
        batch = next(it)
        if(self.heterograph):
            print("HETERO GRAPH IS NOT SUPPORT YET")
            return batch

        else:
            index = batch[0].to(self.gids_device)
            index_size = len(index)
            index_ptr = index.data_ptr()

            return_torch_shape = [index_size,dim]
            return_torch = self.NVshmem_tensor_manager.get_batch_tensor(return_torch_shape)
            return_torch_ptr = self.NVshmem_tensor_manager.get_batch_tensor_ptr()

            request_tensor_ptr = self.NVshmem_tensor_manager.get_index_tensor_ptr()
            self.NVSHMEM_Cache.send_requests(index_ptr, index_size, request_tensor_ptr, self.max_sample_size)
            
            #self.NVshmem_manager.send_requests(index_ptr, index_size, request_tensor_ptr)
            #self.NVshmem_manager.NVshmem_quiet()

            # it = self.NVshmem_tensor_manager.get_index_tensor().to("cpu")
            # print(f"Rank: {self.rank} Request Split: {it}")


            self.NVSHMEM_Cache.dist_read_feature(return_torch_ptr, request_tensor_ptr, self.max_sample_size, dim, self.cache_dim)
            self.GIDS_time += time.time() - GIDS_time_start

            # if(self.rank == 0):
            #     cpu_index = index.to("cpu")
            #     print(f"Rank: {self.rank} Index : {cpu_index}")
            #     return_torch_cpu = return_torch.to("cpu")
            #     print(f"Rank: {self.rank} return torch: {return_torch_cpu[0]}")
            batch = (*batch, return_torch)
            return batch


    #Fetching Data from the SSDs
    def Independent_Cache_fetch_feature(self, dim, it, device):
        GIDS_time_start = time.time()

        batch = next(it)

  
     
        if(self.heterograph):
            print("HETERO GRAPH IS NOT SUPPORT YET")
            return batch

        else:
            index = batch[0].to(self.gids_device)
            index_size = len(index)
            index_ptr = index.data_ptr()

            return_torch_shape = [index_size,dim]
            return_torch = None

            if(self.nvshmem_test):
                return_torch = self.NVshmem_tensor_manager.get_batch_tensor(return_torch_shape)
            else:
                return_torch =  torch.zeros(return_torch_shape, dtype=torch.float, device=self.gids_device).contiguous()


            self.NVSHMEM_Cache.read_feature(return_torch.data_ptr(), index_ptr, index_size, dim, self.cache_dim)
            self.GIDS_time += time.time() - GIDS_time_start
            batch = (*batch, return_torch)
            return batch

    def print_stats(self):
        self.NVSHMEM_Cache.print_stats()


    def __del__(self):
        """
        Destructor to clean up and finalize MPI.
        """
        # Check if NVSHMEM_Cache has been initialized to avoid errors
        if(self.use_nvshmem_tensor or self.nvshmem_test):
            try:
                self.NVSHMEM_Cache.free(self.NVshmem_tensor_manager.nvshmem_batch_ptr)
                self.NVSHMEM_Cache.free(self.NVshmem_tensor_manager.nvshmem_index_ptr)
            except Exception as e:
                print(f"Error during NVShmem Free for rank {self.rank}: {e}")

        try:      
            self.NVSHMEM_Cache.finalize()
            print(f"MPI finalized successfully for rank {self.rank}.")
        except Exception as e:
            print(f"Error during MPI finalization for rank {self.rank}: {e}")
        # if hasattr(self, 'NVshmem_manager'):
        #     try:
        #         # Call finalize_mpi if NVSHMEM_Cache is initialized
        #         self.NVshmem_manager.free(self.NVshmem_tensor_manager.nvshmem_batch_ptr)
        #         self.NVshmem_manager.free(self.NVshmem_tensor_manager.nvshmem_index_ptr)
                
        #         print(f"MPI finalized successfully for rank {self.rank}.")
        #     except Exception as e:
        #         print(f"Error during MPI finalization for rank {self.rank}: {e}")

    

    