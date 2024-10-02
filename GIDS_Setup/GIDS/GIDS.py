import time
import torch
import numpy as np
import ctypes
import nvtx 

import BAM_Feature_Store
from BAM_Feature_Store import Emulate_SA

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


class Emulate_CollateWrapper(object):
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
        graph_device = getattr(self.g, 'device', None)   
        items = recursive_apply(items, lambda x: x.to(self.device))


        if(self.distribute_method == "baseline"):
            item_list = self.baseline_distribute(items)
        # Leverage Training node color
        elif(self.distribute_method == "graph_color"):
            item_list = self.distribute_node(items)
        # Leverage Cache status
        elif(self.distribute_method == "cache_meta"):

            return self.call_cache_meta(items)
        else:
            item_list = self.baseline_distribute(items)


        with nvtx.annotate("Sample", color="green"):
            #batch =  self.sample_func(self.g, items)
            batch = self.sample_func(self.g, item_list[0])

        index_size = len(batch[0])
        index = batch[0].to(self.device)

        return_torch =  torch.zeros([index_size, self.dim], dtype=torch.float, device=self.device)

        static_info_ten = None
        if(self.eviction_policy == 1):
            static_info_ten = self.get_static_info(index)
            self.Emul_SA.read_feature(return_torch.data_ptr(), index.data_ptr(), index_size, self.dim, self.cache_dim, 0, static_info_ten.data_ptr())                 
        else:
            self.Emul_SA.read_feature(return_torch.data_ptr(), index.data_ptr(), index_size, self.dim, self.cache_dim, 0, 0)                 

        #print("batch: ", batch)
        return batch

    def get_static_info(self, index):
        index_cpu = index.cpu()
        static_info_tensor_cpu = self.cache_track.static_info[index_cpu]
        static_info_tensor = static_info_tensor_cpu.to('cuda:0')
        return static_info_tensor

    def call_cache_meta(self, items):
        
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


    def baseline_distribute(self, items):
        N = self.cache_track.num_gpus
        result = [items[i::N] for i in range(N)]
        return result

    def distribute_node(self, items):
        nvlink_batch_size = len(items) / self.cache_track.num_gpus

        distributed_items = []
        for i in range(self.cache_track.num_gpus):
            distributed_items.append([])


        for node_tensor in items:
            node = node_tensor.item()
            nvlink_idx, node_color = self.score_node(node, nvlink_batch_size, distributed_items )
            distributed_items[nvlink_idx].append(node)

            if (node_color != 0 and (self.distribute_method != "cache_meta")):
                if node_color in self.cache_track.cache_track[nvlink_idx]:
                    self.cache_track.cache_track[nvlink_idx][node_color] += 1
                else:
                    self.cache_track.cache_track[nvlink_idx][node_color] = 1

            #print("Dict: ", self.cache_track.cache_track)


        distributed_tensors = []
        for i in range(self.cache_track.num_gpus):
            distributed_tensors.append(torch.tensor(distributed_items[i], device=self.device))
        return distributed_tensors

    def score_node(self, node, batch_size, distributed_items):
        score_tensor = torch.zeros(self.cache_track.num_gpus, dtype=torch.float32)
        color_ten = []
        for nvlink_idx in range(self.cache_track.num_gpus):
            if(len(distributed_items[nvlink_idx]) == batch_size):
                score_tensor[nvlink_idx] = -1.0
                color_ten.append(0)
            
            else:
                node_color = self.cache_track.color_tensor[node].item()
                current_score = 0.0
                if node_color != 0:
                    if(self.distribute_method == "cache_meta"):
                        n_score = self.Emul_SA_list[nvlink_idx].color_score(node_color)
                        current_score += n_score
                    else:
                        if node_color in self.cache_track.cache_track[nvlink_idx]:
                            current_score += self.cache_track.cache_track[nvlink_idx][node_color]
                            #print("new score: ", current_score)
                
                # Need to add adj color matrix
                score_tensor[nvlink_idx] = current_score
                color_ten.append(node_color)
        #print(score_tensor)
        max_idx = torch.argmax(score_tensor)
        return max_idx, color_ten[max_idx]


class _PrefetchingIter(object):
    def __init__(self, dataloader, dataloader_it, GIDS_Loader=None):
        self.dataloader_it = dataloader_it
        self.dataloader = dataloader
        self.graph_sampler = self.dataloader.graph_sampler
        self.GIDS_Loader=GIDS_Loader
        if(self.GIDS_Loader.eviction_policy == 1 or self.GIDS_Loader.eviction_policy == 3 ):
            self.static_info = True
        else:
            self.static_info = False
        
    def __iter__(self):
        return self

    def __next__(self):
        cur_it = self.dataloader_it
        if (self.dataloader.use_ddp):
            batch = self.GIDS_Loader.distributed_fetch_feature(self.dataloader.dim, cur_it, self.GIDS_Loader.gids_device)
        else:
            batch = self.GIDS_Loader.fetch_feature(self.dataloader.dim, cur_it, self.GIDS_Loader.gids_device)
        return batch

class _ID_PrefetchingIter(object):
    def __init__(self, dataloader, dataloader_it):
        self.dataloader_it = dataloader_it
        self.dataloader = dataloader
        self.graph_sampler = self.dataloader.graph_sampler
        
    def __iter__(self):
        return self

    def __next__(self):
        cur_it = self.dataloader_it
        return next(cur_it)


class Cache_Tracker():
    def __init__(self, color_tensor, color_topk, num_gpus):
        self.num_gpus = num_gpus
        self.color_tensor = color_tensor
        self.num_colors = color_topk.shape[0]
        self.color_topk = color_topk
        self.cache_track = []
        self.static_info = None

        for i in range(num_gpus):
            self.cache_track.append({})

    def set_static_info(self, static_info_tensor):
        self.static_info = static_info_tensor


class ID_Loader(torch.utils.data.DataLoader):
    def __init__(self, graph, indices, graph_sampler, batch_size, dim, device=None, use_ddp=False,
                 ddp_seed=0, drop_last=False, shuffle=False,
                 use_alternate_streams=None,
                 num_sets = 64,
                 num_ways = 32,
                 page_size = 4096,
                 cudaDevice = 0,
                 eviction_policy = 0,
                 ##Graph Coloring Metadata
                 color_tensor = None,
                 color_topk = None,
                 num_gpus = 1,
                 distribute_method = "baseline",
                 static_info_tensor = None,
                 **kwargs):


        use_uva = False
        self.GIDS_Loader = GIDS
        self.eviction_policy = eviction_policy
       
        #FIX   
        self.dim = 1024
        self.cache_dim = 1024
        self.Emul_SA_list = []
        self.Emul_SA = None

        self.distribute_method = distribute_method
        self.cache_track = None
        if(color_tensor != None and color_topk != None):
            print("init Cache Tracker")
            self.cache_track = Cache_Tracker(color_tensor, color_topk, num_gpus)
            if(static_info_tensor is not None):
                self.cache_track.set_static_info(static_info_tensor)

        if(distribute_method == "cache_meta"):
            for i in range(num_gpus):
                cur_Emul_SA = Emulate_SA()
                cur_Emul_SA.init_cache(num_sets, num_ways, page_size, cudaDevice, eviction_policy, 1, self.cache_track.num_colors)
                self.Emul_SA_list.append(cur_Emul_SA)
        else:
            self.Emul_SA = Emulate_SA()
            self.Emul_SA.init_cache(num_sets, num_ways, page_size, cudaDevice, eviction_policy, 0, 0)

        if isinstance(kwargs.get('collate_fn', None), Emulate_CollateWrapper):
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
            collate_fn=Emulate_CollateWrapper(
                self.graph_sampler.sample, graph, self.device, self.Emul_SA, self.Emul_SA_list, self.dim, self.cache_dim, self.eviction_policy, self.cache_track, distribute_method),
            batch_size=None,
            pin_memory=False,
            worker_init_fn=worker_init_fn,
            **kwargs)

    def __iter__(self):
        if self.shuffle:
            self.dataset.shuffle()
        num_threads = torch.get_num_threads() if self.num_workers > 0 else None
        return _ID_PrefetchingIter(
            self, super().__iter__())

    def print_counters(self):
        if(self.distribute_method == "cache_meta"):
            self.Emul_SA_list[0].print_counters()
        else:
            self.Emul_SA.print_counters()


class GIDS_DGLDataLoader(torch.utils.data.DataLoader):

    def __init__(self, graph, indices, graph_sampler, batch_size, dim, GIDS, device=None, use_ddp=False,
                 ddp_seed=0, drop_last=False, shuffle=False,
                 use_alternate_streams=None, 
                 
                 **kwargs):

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
        # When using multiprocessing PyTorch sometimes set the number of PyTorch threads to 1
        # when spawning new Python threads.  This drastically slows down pinning features.
        num_threads = torch.get_num_threads() if self.num_workers > 0 else None
        return _PrefetchingIter(
            self, super().__iter__(), GIDS_Loader=self.GIDS_Loader)
 
    def print_stats(self):
        self.GIDS_Loader.print_stats()

    def print_timer(self):
        #if(self.bam):
        #     print("feature aggregation time test: %f" % self.sample_time)
        #print("graph travel time: %f" % self.graph_travel_time)
        self.sample_time = 0.0
        self.graph_travel_time = 0.0


class GIDS():
    def __init__(self, page_size=4096, off=0, cache_dim = 1024, num_ele = 300*1000*1000*1024, 
        num_ssd = 1,  ssd_list = None, set_associative_cache=True, cache_size = 10,  num_ways=4,
        ctrl_idx=0, 
        window_buffer=False,
        accumulator_flag = False, 
        heterograph=False,
        heterograph_map=None,
        device_id=0,
        rank = 0,
        world_size = 1,
        use_ddp = False, 
        fan_out = None, 
        batch_size = 1024,
        wb_size = 8, 
        use_WB = False,
        use_PVP = False,
        pvp_depth = 128, 
        eviction_policy = None,
        static_info_file = None,
        debug_mode = False,
        refresh_time = 1, 
        shared_tensor = None,
        heterogeneous = False, 
        hetero_map=None):
        #self.sample_type = "LADIES"

 
        self.BAM_FS = BAM_Feature_Store.BAM_Feature_Store_float()
        self.heterogeneous = heterogeneous
        self.hetero_map = hetero_map

        if(self.heterogeneous):
            if(hetero_map == None):
                print("Need key-offset map for heterogeneous graph")
        


        
        self.shared_tensor = shared_tensor
        self.shared_tensor_list = []
        self.use_PVP = use_PVP
        self.pvp_depth = pvp_depth
        #DDP parameters
        self.rank = rank
        self.world_size = world_size
        self.use_ddp = use_ddp
        self.index_len_tensor_ptr_list = None
        self.index_len_tensor_list = None
        self.gathered_index_len_tensor_list = None
        self.max_sample_size = batch_size
        self.fan_out = fan_out
        self.debug_mode = debug_mode
        for i in fan_out:
            self.max_sample_size *= i
        print("max sample size: ", self.max_sample_size)

        # CPU Buffer and Storage Access Accumulator Metadata
        self.accumulator_flag = accumulator_flag
        self.required_accesses = 0
        self.prev_cpu_access = 0
        self.return_torch_buffer = []
        self.index_list = []

        # Window Buffering MetaData for Prefetcing
        self.use_WB = use_WB
        #PVP need use_WB
        if(use_PVP):
            self.use_WB = True
        self.wb_size = wb_size
        self.wb_init = False

        self.wb_batch_buffer = []
    
        self.wb_orig_index_size_list = []
        self.wb_gathered_index_size_list = []
        self.wb_gathered_index_list = []
        self.wb_meta_data_list = []
   
        # Window Buffering MetaData
        self.window_buffering_flag = window_buffer
        self.window_buffer = []
        #self.wb_size = wb_size

        # PVP
        self.fill_batch_flag_list = []
        self.refresh_time = refresh_time
        # Cache Parameters
        self.set_associative_cache = set_associative_cache
        self.page_size = page_size
        self.off = off
        self.num_ele = num_ele
        self.cache_size = cache_size
        self.num_ways = num_ways
        self.num_ssd = num_ssd


        # Eviction Policy
        # 0: round robin
        # 1: static
        # 2: dynamic
        # 3: hybrid

        if(eviction_policy == None or eviction_policy == "round_robind"):
            print(" GIDS eviction policy: round_robind")
            self.eviction_policy = 0
    
        elif(eviction_policy == 'static'):
            print(" GIDS eviction policy: static")
            self.eviction_policy = 1
        
        elif(eviction_policy == 'dynamic'):
            print(" GIDS eviction policy: dynamic")
            self.eviction_policy = 2
        
        elif(eviction_policy == 'hybrid'):
            print(" GIDS eviction policy: hybrid")
            self.eviction_policy = 3
        
        else:
            print("Only support 0: round robind, 1: static, 2: dynamic, and 3: hybrid evictio policies")
            self.eviction_policy = 0
        

        #True if the graph is heterogenous graph
        self.heterograph = heterograph
        self.heterograph_map = heterograph_map
        self.graph_GIDS = None

        self.cache_dim = cache_dim
        #self.gids_device="cuda:" + str(ctrl_idx)
        self.gids_device="cuda:" + str(device_id)
        
        self.device_id = device_id
        self.GIDS_controller = BAM_Feature_Store.GIDS_Controllers()

        if (ssd_list == None):
            print("SSD are not assigned")
            self.ssd_list = [i for i in range(self.num_ssd)] 
        else:
            self.ssd_list = ssd_list

        self.GIDS_controller.init_GIDS_controllers(self.num_ssd, 1024, 128, self.ssd_list, device_id)

    

        if(set_associative_cache):
            if(debug_mode):
                self.BAM_FS.init_set_associative_cache(self.GIDS_controller, page_size, off, cache_size, world_size, num_ele, self.num_ssd, num_ways, self.use_WB, self.use_PVP, self.wb_size, self.pvp_depth, self.max_sample_size, self.refresh_time, self.eviction_policy, 1)
            else:
                self.BAM_FS.init_set_associative_cache(self.GIDS_controller, page_size, off, cache_size, world_size, num_ele, self.num_ssd, num_ways, self.use_WB, self.use_PVP, self.wb_size, self.pvp_depth, self.max_sample_size, self.refresh_time, self.eviction_policy, 0)

        else:
            self.BAM_FS.init_controllers(self.GIDS_controller, page_size, off, cache_size,num_ele, self.num_ssd)


        if(self.eviction_policy == 1 or self.eviction_policy == 3):
            if(static_info_file == None):
                print("Need static informatio binary file!")
            else:
                print("Static information file: ", static_info_file)
            self.BAM_FS.create_static_info_buffer(static_info_file)
        


        self.test_flag = False
        if(use_ddp and self.test_flag == False):
            self.index_len_tensor_list = [torch.zeros([1], dtype=torch.int64).to(self.gids_device) for i in range(self.world_size)]
            self.gathered_index_len_tensor_list = [torch.zeros([1], dtype=torch.int64).to(self.gids_device) for i in range(self.world_size)]
            
          

            index_len_tensor_ptr_list = []
            for i in range(self.world_size):
                index_len_tensor_ptr_list.append(self.index_len_tensor_list[i].data_ptr())
            
            self.cpu_index_len_tensor_ptr_list = index_len_tensor_ptr_list
            self.index_len_tensor_ptr_list = torch.tensor(index_len_tensor_ptr_list,dtype=torch.int64).to(self.gids_device).contiguous()

         #            self.BAM_FS.create_meta_buffer(self.world_size, self.max_sample_size)


        # elif(use_ddp and self.test_flag == True):
        #     self.index_len_tensor = torch.zeros([self.world_size], dtype=torch.int64).to(self.gids_device) 
        #     self.gathered_index_len_tensor = torch.zeros([self.world_size], dtype=torch.int64).to(self.gids_device) 
            
            # index_len_tensor_ptr_list = []
            # for i in range(self.world_size):
            #     index_len_tensor_ptr_list.append(self.index_len_tensor_list[i].data_ptr())
            
            # self.cpu_index_len_tensor_ptr_list = index_len_tensor_ptr_list
            # self.index_len_tensor_ptr_list = torch.tensor(index_len_tensor_ptr_list,dtype=torch.int64).to(self.gids_device).contiguous()



    
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

    # For Sampling GIDS operation
    def init_graph_GIDS(self, page_size, off, cache_size, num_ele, num_ssd):
        self.graph_GIDS = BAM_Feature_Store.BAM_Feature_Store_long()        
        self.graph_GIDS.init_controllers(self.GIDS_controller,page_size, off, cache_size, num_ele, num_ssd)

    def get_offset_array(self):
        ret = self.graph_GIDS.get_offset_array()
        return ret

    def get_array_ptr(self):
        return self.graph_GIDS.get_array_ptr()

    # For static CPU feature buffer
    def cpu_backing_buffer(self, dim, length):
        self.BAM_FS.cpu_backing_buffer(dim, length)
        
    def set_cpu_buffer(self, ten, N):
        topk_ten = ten[:N]
        topk_len = len(topk_ten)
        d_ten = topk_ten.to(self.gids_device)
        self.BAM_FS.set_cpu_buffer(d_ten.data_ptr(), topk_len)

    # Window Buffering
    def window_buffering(self, batch):
        s_time = time.time()
        if(self.heterograph):    
             for key, value in batch[0].items():            
                if(len(value) == 0):
                    next
                else:
                    s_time = time.time()
                    input_tensor = value.to(self.gids_device)
                    key_off = 0
                    if(self.heterograph_map != None):
                        if (key in self.heterograph_map):
                            key_off = self.heterograph_map[key]
                        else:
                            print("Cannot find key: ", key, " in the heterograph map!")
                        
                    num_pages = len(input_tensor)
                    self.BAM_FS.set_window_buffering(input_tensor.data_ptr(), num_pages, key_off)
                    e_time = time.time()
                    self.WB_time += e_time - s_time
        
        else:
            input_tensor = batch[0].to(self.gids_device)
            num_pages = len(input_tensor)
            self.BAM_FS.set_window_buffering(input_tensor.data_ptr(), num_pages, 0)
            e_time = time.time()
            self.WB_time += e_time - s_time
            

    # Window Buffering Helper Function    
    def fill_wb(self, it, num):
        for i in range(num):
            batch = next(it)
            self.window_buffer.append(batch)
            #run window buffering for the current batch
            self.window_buffering(batch)
        

    # BW in GB/s, latency in micro seconds
    def set_required_storage_access(self, bw, l_ssd, l_system, num_ssd, p):
        accesses = (p * bw * 1024 / self.page_size * (l_ssd + l_system) * num_ssd) / (1-p)
        self.required_accesses = accesses
        print("Number of required storage accesses: ", accesses)

    #Fetching Data from the SSDs
    def fetch_feature(self, dim, it, device):
        if(self.set_associative_cache):
            batch =  self.SA_fetch_feature(dim, it, device)
        else:
            batch = self.DM_fetch_feature(dim, it, device)
        return batch

    def DM_fetch_feature(self, dim, it, device):
        GIDS_time_start = time.time()
        if(self.window_buffering_flag):
            #Filling up the window buffer
            if(self.wb_init == False):
                self.fill_wb(it, self.wb_size)
                self.wb_init = True

        #print("Sample  start")
        next_batch = next(it)
        #print("Sample  done")

        self.window_buffer.append(next_batch)
        #Update Counters for Windwo Buffering
        if(self.window_buffering_flag):
            self.window_buffering(next_batch)
        
        # When the Storage Access Accumulator is enabled
        if(self.accumulator_flag):
            index_size_list = []
            index_ptr_list = []
            return_torch_list = []
            key_list = []

            if(len(self.return_torch_buffer) != 0):
                return_ten = self.return_torch_buffer.pop(0)
                return_batch = self.window_buffer.pop(0)
                return_batch.append(return_ten)
                self.GIDS_time += time.time() - GIDS_time_start
                return return_batch

            buffer_size = len(self.window_buffer)
            current_access = 0
            num_iter = 0
            required_accesses = self.required_accesses


            if(self.heterograph):
                while(1):
                    if(num_iter >= buffer_size):
                        batch = next(it)
                        for k , v in batch[0].items():
                            current_access += len(v)
                        
                        self.window_buffer.append(batch)
                        if(self.window_buffering_flag):
                            self.window_buffering(batch)

                    else:
                        batch = self.window_buffer[num_iter]
                        for k , v in batch[0].items():
                            current_access += len(v)

                    num_iter +=1
                    required_accesses += self.prev_cpu_access
                    if(current_access > (required_accesses )):
                        break

                num_concurrent_iter = 0
                for i in range(num_iter):
                    batch = self.window_buffer[i]
                    ret_ten = {}
                    for k , v in batch[0].items():
                        if(len(v) == 0):
                            empty_t = torch.empty((0, dim)).to(self.gids_device)
                            ret_ten[k] = empty_t
                        else:
                            key_off = 0
                            if(self.heterograph_map != None):
                                if (key in self.heterograph_map):
                                    key_off = self.heterograph_map[key]
                                else:
                                    print("Cannot find key: ", key, " in the heterograph map!")
                            v = v.to(self.gids_device)
                            index_size = len(v)
                            index_size_list.append(index_size)
                            return_torch =  torch.zeros([index_size,dim], dtype=torch.float, device=self.gids_device)
                            index_ptr_list.append(v.data_ptr())
                            ret_ten[k] = return_torch
                            return_torch_list.append(return_torch.data_ptr())
                            key_list.append(key_off)
                            num_concurrent_iter += 1
                    self.return_torch_buffer.append(ret_ten)
                self.BAM_FS.read_feature_merged_hetero(num_concurrent_iter, return_torch_list, index_ptr_list, index_size_list, dim, self.cache_dim, key_list)

                return_ten = self.return_torch_buffer.pop(0)
                return_batch = self.window_buffer.pop(0)
                return_batch.append(return_ten)
                self.GIDS_time += time.time() - GIDS_time_start

                cpu_access_count = self.BAM_FS.get_cpu_access_count()
                self.prev_cpu_access = int(cpu_access_count / num_iter)
                self.BAM_FS.flush_cpu_access_count()

                return return_batch
            else:
                while(1):
                    if(num_iter >= buffer_size):
                        batch = next(it)
                        current_access += len(batch[0])
                        self.window_buffer.append(batch)
                        if(self.window_buffering_flag):
                            self.window_buffering(batch)
                    else:
                        batch = self.window_buffer[num_iter]
                        current_access += len(batch[0])
                    num_iter +=1
                    required_accesses += self.prev_cpu_access
                    if(current_access > (required_accesses )):
                        break

                for i in range(num_iter):
                    batch = self.window_buffer[i]
                    batch[0] = batch[0].to(self.gids_device)
                    index_size = len(batch[0])
                    index_size_list.append(index_size)
                    return_torch =  torch.zeros([index_size,dim], dtype=torch.float, device=self.gids_device)
                    index_ptr_list.append(batch[0].data_ptr())
                    return_torch_list.append(return_torch.data_ptr())
                    self.return_torch_buffer.append(return_torch)

                self.BAM_FS.read_feature_merged(num_iter, return_torch_list, index_ptr_list, index_size_list, dim, self.cache_dim)
                return_ten = self.return_torch_buffer.pop(0)
                return_batch = self.window_buffer.pop(0)
                return_batch.append(return_ten)
                self.GIDS_time += time.time() - GIDS_time_start

                cpu_access_count = self.BAM_FS.get_cpu_access_count()
                self.prev_cpu_access = int(cpu_access_count / num_iter)
                self.BAM_FS.flush_cpu_access_count()

                return return_batch
        
        # Storage Access Accumulator is disabled
        else:
            if(self.heterograph):
                batch = self.window_buffer.pop(0)
                ret_ten = {}
                index_size_list = []
                index_ptr_list = []
                return_torch_list = []
                key_list = []
                
                num_keys = 0
                for key , v in batch[0].items():
                    if(len(v) == 0):
                        empty_t = torch.empty((0, dim)).to(self.gids_device).contiguous()
                        ret_ten[key] = empty_t
                    else:
                        key_off = 0
                        if(self.heterograph_map != None):
                            if (key in self.heterograph_map):
                                key_off = self.heterograph_map[key]
                            else:
                                print("Cannot find key: ", key, " in the heterograph map!")
                        
                        g_index = v.to(self.gids_device)
                        index_size = len(g_index)
                        index_ptr = g_index.data_ptr()
                        
                        return_torch =  torch.zeros([index_size,dim], dtype=torch.float, device=self.gids_device).contiguous()
                        return_torch_list.append(return_torch.data_ptr())
                        ret_ten[key] = return_torch
                        num_keys += 1
                        index_ptr_list.append(index_ptr)
                        index_size_list.append(index_size)
                        key_list.append(key_off)

                self.BAM_FS.read_feature_hetero(num_keys, return_torch_list, index_ptr_list, index_size_list, dim, self.cache_dim, key_list)

                batch.append(ret_ten)
                self.GIDS_time += time.time() - GIDS_time_start
                return batch

            else:
                batch = self.window_buffer.pop(0)
                #print("batch 0: ", batch.ndata['_ID'])
                index = batch[0].to(self.gids_device)
                index_size = len(index)
                #print(batch[0])
                index_ptr = index.data_ptr()
                return_torch =  torch.zeros([index_size,dim], dtype=torch.float, device=self.gids_device).contiguous()
                self.BAM_FS.read_feature(return_torch.data_ptr(), index_ptr, index_size, dim, self.cache_dim, 0)
                self.GIDS_time += time.time() - GIDS_time_start

                batch.append(return_torch)

                return batch

    #Fetching Data from the SSDs
    def SA_fetch_feature(self, dim, it, device):
        GIDS_time_start = time.time()

        if(self.window_buffering_flag):
            #Filling up the window buffer
            if(self.wb_init == False):
                self.fill_wb(it, self.wb_size)
                self.wb_init = True

        next_batch = next(it)

        self.window_buffer.append(next_batch)
        #Update Counters for Windwo Buffering
        if(self.window_buffering_flag):
            self.window_buffering(next_batch)
        
     
        if(self.heterograph):
            batch = self.window_buffer.pop(0)
            ret_ten = {}
            index_size_list = []
            index_ptr_list = []
            return_torch_list = []
            key_list = []
            
            num_keys = 0
            for key , v in batch[0].items():
                if(len(v) == 0):
                    empty_t = torch.empty((0, dim)).to(self.gids_device).contiguous()
                    ret_ten[key] = empty_t
                else:
                    key_off = 0
                    if(self.heterograph_map != None):
                        if (key in self.heterograph_map):
                            key_off = self.heterograph_map[key]
                        else:
                            print("Cannot find key: ", key, " in the heterograph map!")
                    
                    g_index = v.to(self.gids_device)
                    index_size = len(g_index)
                    index_ptr = g_index.data_ptr()
                    
                    return_torch =  torch.zeros([index_size,dim], dtype=torch.float, device=self.gids_device).contiguous()
                    return_torch_list.append(return_torch.data_ptr())
                    ret_ten[key] = return_torch
                    num_keys += 1
                    index_ptr_list.append(index_ptr)
                    index_size_list.append(index_size)
                    key_list.append(key_off)

            self.BAM_FS.read_feature_hetero(num_keys, return_torch_list, index_ptr_list, index_size_list, dim, self.cache_dim, key_list)

            batch.append(ret_ten)
            self.GIDS_time += time.time() - GIDS_time_start
            return batch

        else:
            batch = self.window_buffer.pop(0)
            index = batch[0].to(self.gids_device)
            index_size = len(index)
            index_ptr = index.data_ptr()
            return_torch =  torch.zeros([index_size,dim], dtype=torch.float, device=self.gids_device).contiguous()

            

            static_info_ten = None
            static_info_data_ptr = 0
            if(self.eviction_policy == 1 or self.eviction_policy == 3):
                static_info_ten = self.get_static_info(batch)
                static_info_data_ptr = static_info_ten.data_ptr()



            self.BAM_FS.SA_read_feature(return_torch.data_ptr(), index_ptr, index_size, dim, self.cache_dim, 0, static_info_data_ptr)

            self.GIDS_time += time.time() - GIDS_time_start

            #batch.append(return_torch)
            batch = (*batch, return_torch)
            return batch

    #Fetching Data from the SSDs
    def distributed_fetch_feature(self, dim, it, device):
        if(self.heterogeneous):
            if(self.use_WB):
                batch =  self.dist_SA_fetch_feature_WB_hetero(dim, it, device)
            else:
                batch =  self.dist_SA_fetch_feature_hetero(dim, it, device)

        else:
            if(self.use_WB):
                batch =  self.dist_SA_fetch_feature_WB(dim, it, device)
            else:
                batch =  self.dist_SA_fetch_feature(dim, it, device)
        return batch



    def create_ptr_list_tensor(self, tensor_list):
        pointer_list = [tensor.data_ptr() for tensor in tensor_list]
        pointer_tensor = torch.tensor(pointer_list, dtype=torch.int64, device=self.gids_device).contiguous()        
        return pointer_tensor



    def create_ptr_list(self, tensor_list):
        pointer_list = [tensor.data_ptr() for tensor in tensor_list]
        return pointer_list

    def create_ptr_list_2D(self, tensor_list):
        pointer_list = []
        for cur_tensor in tensor_list:
            cur_list =  [tensor.data_ptr() for tensor in cur_tensor]
            pointer_list.extend(cur_list)

    def create_ptr_list_tensor_2D(self, tensor_list, device, skip = 0):
        #print("GPU: ", device, "GIDS device: ", self.gids_device, "World Size: ", self.world_size,"\t\t create_ptr_list_tensor_2D Start tensor list:", tensor_list)

        pointer_list = []
        counter = 0
        for cur_tensor in tensor_list:
            if(counter < skip):
                counter += 1
                continue
            cur_list =  [tensor.data_ptr() for tensor in cur_tensor]
            pointer_list.extend(cur_list)
        
        #print("GPU: ", device, "World Size: ", self.world_size,"\t\t create_ptr_list_tensor_2D For loop Done")

        #pointer_tensor = torch.tensor(pointer_list, dtype=torch.int64, device=self.gids_device).contiguous()     
        pointer_tensor = torch.tensor(pointer_list, dtype=torch.int64, device=self.gids_device).contiguous()     

        #print("GPU: ", device, "World Size: ", self.world_size,"\t\t create_ptr_list_tensor_2D For Tensor Done")
   
        return pointer_tensor

    def create_tensor_from_list_2D(self, tensor_list, skip = 0):
        data_list = []
        counter = 0
        for cur_tensor in tensor_list:
            if(counter < skip):
                counter += 1
                continue
            for tensor in cur_tensor:
                data_list.append(tensor)
        
        pointer_tensor = torch.tensor(data_list, dtype=torch.int64, device=self.gids_device).contiguous()        
        return pointer_tensor

    

    # index = [index for the current batch]
    # index_len_tensor_ptr_list = [idx for GPU0, idx for GPU1, ....]    
    # my_bucket_list = [tensor idx for GPU0, tensor idx for GPU1, ...]


    def split_index_tensor(self, index, my_bucket_list, split_len_list, meta_data_list):
        split_start = time.time()
        index_size = len(index)   
        self.BAM_FS.reset_node_counter(self.cpu_index_len_tensor_ptr_list, self.world_size)

        self.BAM_FS.split_node_list_init(index.data_ptr(), self.world_size, index_size, self.index_len_tensor_ptr_list.data_ptr())

        
        for i in range(self.world_size):
            bucket_len = self.index_len_tensor_list[i].to("cpu").item()

            bucket_torch = torch.zeros([bucket_len], dtype=torch.int64, device=self.gids_device).contiguous()
            meta_torch = torch.zeros([bucket_len], dtype=torch.int64, device=self.gids_device).contiguous()

            my_bucket_list.append(bucket_torch)
            split_len_list.append(bucket_len)
            meta_data_list.append(meta_torch)

        my_bucket_ptr_list_tensor = self.create_ptr_list_tensor(my_bucket_list)
        meta_data_list_tensor = self.create_ptr_list_tensor(meta_data_list)
        
        self.BAM_FS.reset_node_counter(self.cpu_index_len_tensor_ptr_list, self.world_size)
        self.BAM_FS.split_node_list(index.data_ptr(), self.world_size, index_size, my_bucket_ptr_list_tensor.data_ptr(), self.index_len_tensor_ptr_list.data_ptr(), meta_data_list_tensor.data_ptr())

        self.Split_time += (time.time() - split_start)



    def split_index_tensor2(self, index, my_bucket_list, split_len_list, meta_data_list):
        split_start = time.time()
        index_size = len(index)   

        self.BAM_FS.reset_node_counter2(self.index_len_tensor.data_ptr(), self.world_size)
        self.BAM_FS.split_node_list_init2(index.data_ptr(), self.world_size, index_size, self.index_len_tensor.data_ptr())

        cpu_ten =  self.index_len_tensor.to("cpu")
        for i in range(self.world_size):
            bucket_len = cpu_ten[i].item()

            bucket_torch = torch.zeros([bucket_len], dtype=torch.int64, device=self.gids_device).contiguous()
            meta_torch = torch.zeros([bucket_len], dtype=torch.int64, device=self.gids_device).contiguous()

            my_bucket_list.append(bucket_torch)
            split_len_list.append(bucket_len)
            meta_data_list.append(meta_torch)

        my_bucket_ptr_list_tensor = self.create_ptr_list_tensor(my_bucket_list)
        meta_data_list_tensor = self.create_ptr_list_tensor(meta_data_list)
        
        self.BAM_FS.reset_node_counter2(self.index_len_tensor.data_ptr(), self.world_size)
        self.BAM_FS.split_node_list(index.data_ptr(), self.world_size, index_size, my_bucket_ptr_list_tensor.data_ptr(), self.index_len_tensor.data_ptr(), meta_data_list_tensor.data_ptr())

        self.Split_time += (time.time() - split_start)


    # dist_list is updated
    @nvtx.annotate("Gather Tensor Func", color="green")
    def gather_tensors(self, dst_list, src_list):
        com_start = time.time()
        for i in range(self.world_size):
            #Sender
            if(self.rank == i):
                for j in range(self.world_size):
                    if(self.rank != j):
                        dist.send(tensor=src_list[j], dst=j)
                    else:
                        dst_list[j] = src_list[j]
            #Receiver
            else:
                dist.recv(tensor=dst_list[i], src=i)

        self.Communication_time += (time.time() - com_start)
        return

    @nvtx.annotate("Gather Tensor AsyncFunc", color="green")
    def gather_tensors_async(self, dst_list, src_list):
        com_start = time.time()
        requests = []

        for i in range(self.world_size):
            # Sender
            if self.rank == i:
                for j in range(self.world_size):
                    if self.rank != j:
                        req = dist.isend(tensor=src_list[j], dst=j)
                        requests.append(req)
                    else:
                        dst_list[j] = src_list[j]
            # Receiver
            else:
                req = dist.irecv(tensor=dst_list[i], src=i)
                requests.append(req)

        # Wait for all communications to complete
        for req in requests:
            req.wait()

        torch.distributed.barrier()

        self.Communication_time += (time.time() - com_start)
        return


    def communication_setup(self, dim, orig_index_size_list, gathered_index_list, gathered_index_size_list):
        setup_start = time.time()
        for i in range(self.world_size):
            #cur_len = self.gathered_index_len_tensor_list[i].to("cpu")

    
            cur_len = self.shared_tensor[i][self.rank]

            #print("Rank: ", self.rank, "shared cur len: ", cur_len1, "cur len: ", cur_len)


            gathered_index_size_list.append(cur_len.item())

            cur_index_torch = torch.zeros([cur_len], dtype=torch.int64, device=self.gids_device).contiguous()
            gathered_index_list.append(cur_index_torch)

            recv_len = self.index_len_tensor_list[i].to("cpu")
            orig_index_size_list.append(recv_len) 
        self.Communication_setup_time += (time.time() - setup_start)

    def communication_setup2(self, dim, orig_index_size_list, gathered_index_list, gathered_index_size_list):
        setup_start = time.time()
        for i in range(self.world_size):
            cur_len = self.gathered_index_len_tensor[i].to("cpu")


            gathered_index_size_list.append(cur_len.item())

            cur_index_torch = torch.zeros([cur_len], dtype=torch.int64, device=self.gids_device).contiguous()
            gathered_index_list.append(cur_index_torch)

            recv_len = self.index_len_tensor[i].to("cpu")
            orig_index_size_list.append(recv_len) 
        self.Communication_setup_time += (time.time() - setup_start)

    @nvtx.annotate("Alloc Tensor", color="red")
    def alloc_tensors(self, dim, orig_index_size_list, orig_tensor_list, gathered_index_size_list, gathered_tensor_list):
        for i in range(self.world_size):
            orig_tensor_len = orig_index_size_list[i]
            gathered_tensor_len = gathered_index_size_list[i]
            orig_tensor =  torch.zeros([orig_tensor_len,dim], dtype=torch.float, device=self.gids_device).contiguous()
            gathered_tensor =  torch.zeros([gathered_tensor_len,dim], dtype=torch.float, device=self.gids_device).contiguous()
            orig_tensor_list.append(orig_tensor)
            gathered_tensor_list.append(gathered_tensor)

    def alloc_flag_tensors(self, dim, input_size_list, output_tensor_list):
        for i in range(len(input_size_list)):
            batch_len = input_size_list[i].item()

         #   print("batch len: ", batch_len)
            node_flag_tensor =  torch.zeros(batch_len, dtype=torch.int64, device=self.gids_device).contiguous()
            output_tensor_list.append(node_flag_tensor)
        #    self.alloc_flag_tensors(dim, gathered_index_size_list, fill_batch_flag_list)



        


    def distribute_index(self, dim, it, device):
        sample_start = time.time()
        batch = next(it)
        self.Sampling_time += time.time() - sample_start

        index = batch[0].to(self.gids_device)
        index_size = len(index)   

        orig_index_size_list = []
        gathered_index_size_list = []
        gathered_index_list = []
        meta_data_list = []

        my_bucket_list = []
        split_tensor_len = []

        self.split_index_tensor(index, my_bucket_list, split_tensor_len, meta_data_list)

        torch.cuda.synchronize()
        gather_start = time.time()
        with nvtx.annotate("Init Broadcast2", color="red"):
            for i in range(self.world_size):
                cur_list = self.index_len_tensor_list[i].item()
               # print("rank: ", self.rank, " i: ", i, "cur len : ", cur_list)
                self.shared_tensor[self.rank][i] = cur_list

        torch.distributed.barrier()
        self.Communication_time += (time.time() - gather_start)

    

        with nvtx.annotate("Comm setup", color="blue"):
            self.communication_setup(dim, orig_index_size_list, gathered_index_list, gathered_index_size_list)
        
        with nvtx.annotate("Broadcast Node IDs", color="green"):
            self.gather_tensors(gathered_index_list, my_bucket_list)


        # print("GPU: ", device, "Gather2 done")
        self.wb_batch_buffer.append(batch)
        self.wb_orig_index_size_list.append(orig_index_size_list)
        self.wb_gathered_index_size_list.append(gathered_index_size_list)
        self.wb_gathered_index_list.append(gathered_index_list)
        self.wb_meta_data_list.append(meta_data_list)
        torch.cuda.synchronize()

        
    def init_WB(self, dim, it, device):
        #print( "World Size: ", self.world_size, " WB size: ", self.wb_size)
        for i in range(self.wb_size + 1):
            self.distribute_index(dim, it, device)
        self.wb_init = True

    def init_WB_hetero(self, dim, it, device):
        #print( "World Size: ", self.world_size, " WB size: ", self.wb_size)
        for i in range(self.wb_size + 1):
            self.distribute_index_hetero(dim, it, device)
        self.wb_init = True

    def dist_SA_fetch_feature_WB(self, dim, it, device):
        GIDS_time_start = time.time()
        if(self.wb_init == False):
            self.init_WB(dim, it, device)

        self.distribute_index(dim, it, device)

        batch = self.wb_batch_buffer.pop(0)
        orig_index_size_list = self.wb_orig_index_size_list.pop(0)
        gathered_index_size_list = self.wb_gathered_index_size_list.pop(0)
        gathered_index_list = self.wb_gathered_index_list.pop(0)
        meta_data_list = self.wb_meta_data_list.pop(0)
        index_size = len(batch[0]) 
        
        torch.cuda.synchronize()

    
        
        with nvtx.annotate("Fill Batch", color="green"):
            self.BAM_FS.fill_batch()


        # Feature Aggregation
        orig_tensor_list = []
        gathered_tensor_list = []
        self.alloc_tensors(dim, orig_index_size_list, orig_tensor_list, gathered_index_size_list, gathered_tensor_list)

        feature_read_start = time.time()
        index_ptr_list = self.create_ptr_list(gathered_index_list)
        gathered_torch_ptr_list = self.create_ptr_list(gathered_tensor_list)

        static_info_ten_list = []
        staic_info_ten_ptr_list = []
        if(self.eviction_policy == 1 or self.eviction_policy == 3):
            static_info_ten_list, staic_info_ten_ptr_list = self.get_static_info_dist(index_ptr_list, gathered_index_size_list)

        with nvtx.annotate("Feature Aggregation", color="blue"):
            self.BAM_FS.SA_read_feature_dist(gathered_torch_ptr_list, index_ptr_list, gathered_index_size_list, self.world_size, dim, self.cache_dim, 0, staic_info_ten_ptr_list)
            
        self.agg_time += time.time() - feature_read_start
        #Barrier
        torch.distributed.barrier()

        # Gathering Minibatch Tensors
        self.gather_tensors_async(orig_tensor_list, gathered_tensor_list)

        # GPU to GPU memcpy
        gather_start = time.time()
        return_torch =  torch.zeros([index_size,dim], dtype=torch.float, device=self.gids_device).contiguous()

        dist_gather_ptr_list = self.create_ptr_list(orig_tensor_list)
        meta_ptr_list = self.create_ptr_list(meta_data_list)
        
        with nvtx.annotate("Gather Tensor", color="blue"):
            self.BAM_FS.gather_feature_list(return_torch.data_ptr(),dist_gather_ptr_list, orig_index_size_list, self.world_size, dim,  self.rank, meta_ptr_list)
        self.Gather_time += (time.time()) - gather_start

        reset_start = time.time()
        self.BAM_FS.reset_node_counter(self.cpu_index_len_tensor_ptr_list, self.world_size)
        self.Reset_time += (time.time()) - reset_start

        self.GIDS_time += time.time() - GIDS_time_start
        batch2 = (*batch, return_torch) 

        with nvtx.annotate("Prefetching", color="red"):
            self.BAM_FS.prefetch_from_victim_queue()

        # Updating Reuse Value
        wb_index_list_tensor = self.create_ptr_list_tensor_2D(self.wb_gathered_index_list, device, 1)
        wb_size_list_tensor = self.create_tensor_from_list_2D(self.wb_gathered_index_size_list, 1)

        with nvtx.annotate("Update Counters", color="red"):
            self.BAM_FS.update_reuse_counters(wb_index_list_tensor.data_ptr(), wb_size_list_tensor.data_ptr(), self.max_sample_size, self.world_size, self.wb_size)

 
        return batch2
    
    def dist_SA_fetch_feature(self, dim, it, device):
        GIDS_time_start = time.time()

        self.distribute_index(dim, it, device)

        batch = self.wb_batch_buffer.pop(0)
        orig_index_size_list = self.wb_orig_index_size_list.pop(0)
        gathered_index_size_list = self.wb_gathered_index_size_list.pop(0)
        gathered_index_list = self.wb_gathered_index_list.pop(0)
        meta_data_list = self.wb_meta_data_list.pop(0)
        index_size = len(batch[0]) 

        orig_tensor_list = []
        gathered_tensor_list = []
        self.alloc_tensors(dim, orig_index_size_list, orig_tensor_list, gathered_index_size_list, gathered_tensor_list)

        feature_read_start = time.time()
        index_ptr_list = self.create_ptr_list(gathered_index_list)
        gathered_torch_ptr_list = self.create_ptr_list(gathered_tensor_list)
        
        torch.cuda.synchronize()
        
        return_torch2 =  torch.zeros([index_size,dim], dtype=torch.float, device=self.gids_device).contiguous()


        static_info_ten_list = []
        staic_info_ten_ptr_list = []
        if(self.eviction_policy == 1 or self.eviction_policy == 3):
            static_info_ten_list, staic_info_ten_ptr_list = self.get_static_info_dist(index_ptr_list, gathered_index_size_list)

        with nvtx.annotate("Feature Aggregation", color="blue"):
            self.BAM_FS.SA_read_feature_dist(gathered_torch_ptr_list, index_ptr_list, gathered_index_size_list, self.world_size, dim, self.cache_dim, 0, staic_info_ten_ptr_list)
    

        self.agg_time += time.time() - feature_read_start

        # Gathering Minibatch Tensors
        self.gather_tensors_async(orig_tensor_list, gathered_tensor_list)


        # GPU to GPU memcpy
        with nvtx.annotate("GPU to GPU Memcpy", color="green"):

            gather_start = time.time()
            return_torch =  torch.zeros([index_size,dim], dtype=torch.float, device=self.gids_device).contiguous()

            dist_gather_ptr_list = self.create_ptr_list(orig_tensor_list)
            meta_ptr_list = self.create_ptr_list(meta_data_list)
            self.BAM_FS.gather_feature_list(return_torch.data_ptr(),dist_gather_ptr_list, orig_index_size_list, self.world_size, dim,  self.rank, meta_ptr_list)
            self.Gather_time += (time.time()) - gather_start

            reset_start = time.time()
            self.BAM_FS.reset_node_counter(self.cpu_index_len_tensor_ptr_list, self.world_size)
            self.Reset_time += (time.time()) - reset_start

            self.GIDS_time += time.time() - GIDS_time_start        
        batch2 = (*batch, return_torch) 

        return batch2



# Heterogeneous Graph

    def split_index_tensor_hetero(self, index, my_bucket_list, split_len_list, meta_data_list):
        split_start = time.time()
        self.BAM_FS.reset_node_counter(self.cpu_index_len_tensor_ptr_list, self.world_size)
        
        index_data_ptr_list = []
        index_size_list = []
        
        for key, tensor in index.items():
            ten_len = tensor.size()[0]
            if(ten_len != 0):
                index_size_list.append(tensor.size()[0])
                index_data_ptr_list.append(tensor.data_ptr())

        self.BAM_FS.split_node_list_init_hetero(index_data_ptr_list, self.world_size, index_size_list, self.index_len_tensor_ptr_list.data_ptr())

        
        for i in range(self.world_size):
            bucket_len = self.index_len_tensor_list[i].to("cpu").item()
            bucket_torch = torch.zeros([bucket_len], dtype=torch.int64, device=self.gids_device).contiguous()
            meta_torch = torch.zeros([bucket_len], dtype=torch.int64, device=self.gids_device).contiguous()

            my_bucket_list.append(bucket_torch)
            split_len_list.append(bucket_len)
            meta_data_list.append(meta_torch)

        my_bucket_ptr_list_tensor = self.create_ptr_list_tensor(my_bucket_list)
        meta_data_list_tensor = self.create_ptr_list_tensor(meta_data_list)
        
        self.BAM_FS.reset_node_counter(self.cpu_index_len_tensor_ptr_list, self.world_size)

        self.BAM_FS.split_node_list_hetero(index_data_ptr_list, self.world_size, index_size_list, my_bucket_ptr_list_tensor.data_ptr(), self.index_len_tensor_ptr_list.data_ptr(), meta_data_list_tensor.data_ptr())

        self.Split_time += (time.time() - split_start)


    def distribute_index_hetero(self, dim, it, device):
        sample_start = time.time()
        batch = next(it)
        self.Sampling_time += time.time() - sample_start

        index_dict = batch[0]
 
        orig_index_size_list = []
        gathered_index_size_list = []
        gathered_index_list = []
        meta_data_list = []

        my_bucket_list = []
        split_tensor_len = []

        self.split_index_tensor_hetero(index_dict, my_bucket_list, split_tensor_len, meta_data_list)

        # # Gather Node list counter
        # for i in range(self.world_size):
        #     cur_list = None
        #     if(self.rank == i):
        #         cur_list = self.gathered_index_len_tensor_list
        #     dist.gather(self.index_len_tensor_list[i], cur_list, dst=i)
        torch.cuda.synchronize()
        gather_start = time.time()

        with nvtx.annotate("Init Broadcast2", color="red"):
            for i in range(self.world_size):
                cur_list = self.index_len_tensor_list[i].item()
               # print("rank: ", self.rank, " i: ", i, "cur len : ", cur_list)
                self.shared_tensor[self.rank][i] = cur_list
        
        torch.distributed.barrier()
        self.Communication_time += (time.time() - gather_start)

        self.communication_setup(dim, orig_index_size_list, gathered_index_list, gathered_index_size_list)
        self.gather_tensors(gathered_index_list, my_bucket_list)


        # print("GPU: ", device, "Gather2 done")
        self.wb_batch_buffer.append(batch)
        self.wb_orig_index_size_list.append(orig_index_size_list)
        self.wb_gathered_index_size_list.append(gathered_index_size_list)
        self.wb_gathered_index_list.append(gathered_index_list)
        self.wb_meta_data_list.append(meta_data_list)
        torch.cuda.synchronize()



    def dist_SA_fetch_feature_hetero(self, dim, it, device):

        GIDS_time_start = time.time()

        self.distribute_index_hetero(dim, it, device)

        batch = self.wb_batch_buffer.pop(0)
        orig_index_size_list = self.wb_orig_index_size_list.pop(0)
        gathered_index_size_list = self.wb_gathered_index_size_list.pop(0)
        gathered_index_list = self.wb_gathered_index_list.pop(0)
        meta_data_list = self.wb_meta_data_list.pop(0)



        # Feature Aggregation
        orig_tensor_list = []
        gathered_tensor_list = []
        self.alloc_tensors(dim, orig_index_size_list, orig_tensor_list, gathered_index_size_list, gathered_tensor_list)

        feature_read_start = time.time()
        index_ptr_list = self.create_ptr_list(gathered_index_list)
        gathered_torch_ptr_list = self.create_ptr_list(gathered_tensor_list)
        
        torch.cuda.synchronize()
        


        static_info_ten_list = []
        staic_info_ten_ptr_list = []
        if(self.eviction_policy == 1 or self.eviction_policy == 3):
            static_info_ten_list, staic_info_ten_ptr_list = self.get_static_info_dist(index_ptr_list, gathered_index_size_list)

        with nvtx.annotate("Feature Aggregation", color="blue"):
            self.BAM_FS.SA_read_feature_dist(gathered_torch_ptr_list, index_ptr_list, gathered_index_size_list, self.world_size, dim, self.cache_dim, 0, staic_info_ten_ptr_list)
    

        self.agg_time += time.time() - feature_read_start

        # Gathering Minibatch Tensors
        self.gather_tensors(orig_tensor_list, gathered_tensor_list)


        # GPU to GPU memcpy
        with nvtx.annotate("GPU to GPU Memcpy", color="green"):
            return_torch = {}
            gather_start = time.time()
            return_torch_ptr_list = []
            for k , v in batch[0].items():
                if(len(v) == 0):
                        empty_t = torch.empty((0, dim)).to(self.gids_device)
                        return_torch[k] = empty_t
                else:
                    key_off = 0
                    if (k in self.hetero_map):
                            key_off = self.hetero_map[k]
                    else:
                        print("Cannot find key: ", k, " in the heterograph map!")
                    
                    return_ten =  torch.zeros([len(v),dim], dtype=torch.float, device=self.gids_device)
                    return_torch[k] = return_ten
                    return_torch_ptr_list.append(return_ten.data_ptr())



            dist_gather_ptr_list = self.create_ptr_list(orig_tensor_list)
            meta_ptr_list = self.create_ptr_list(meta_data_list)



            return_torch_ptr_ten = torch.tensor(return_torch_ptr_list, dtype=torch.int64, device=self.gids_device).contiguous()     

            #self.BAM_FS.gather_feature_list(return_torch.data_ptr(),dist_gather_ptr_list, orig_index_size_list, self.world_size, dim,  self.rank, meta_ptr_list)
            self.BAM_FS.gather_feature_list_hetero(return_torch_ptr_ten.data_ptr(), dist_gather_ptr_list, orig_index_size_list, self.world_size, dim,  self.rank, meta_ptr_list)

            self.Gather_time += (time.time()) - gather_start

            reset_start = time.time()
            self.BAM_FS.reset_node_counter(self.cpu_index_len_tensor_ptr_list, self.world_size)
            self.Reset_time += (time.time()) - reset_start

            self.GIDS_time += time.time() - GIDS_time_start        
        batch2 = (*batch, return_torch) 

        return batch2

    def dist_SA_fetch_feature_WB_hetero(self, dim, it, device):

        GIDS_time_start = time.time()
        
        if(self.wb_init == False):
            self.init_WB_hetero(dim, it, device)

        self.distribute_index_hetero(dim, it, device)

        batch = self.wb_batch_buffer.pop(0)
        orig_index_size_list = self.wb_orig_index_size_list.pop(0)
        gathered_index_size_list = self.wb_gathered_index_size_list.pop(0)
        gathered_index_list = self.wb_gathered_index_list.pop(0)
        meta_data_list = self.wb_meta_data_list.pop(0)


        with nvtx.annotate("Fill Batch", color="green"):
            self.BAM_FS.fill_batch()

        # Feature Aggregation
        orig_tensor_list = []
        gathered_tensor_list = []
        self.alloc_tensors(dim, orig_index_size_list, orig_tensor_list, gathered_index_size_list, gathered_tensor_list)

        feature_read_start = time.time()
        index_ptr_list = self.create_ptr_list(gathered_index_list)
        gathered_torch_ptr_list = self.create_ptr_list(gathered_tensor_list)
        
        torch.cuda.synchronize()
        


        static_info_ten_list = []
        staic_info_ten_ptr_list = []
        if(self.eviction_policy == 1 or self.eviction_policy == 3):
            static_info_ten_list, staic_info_ten_ptr_list = self.get_static_info_dist(index_ptr_list, gathered_index_size_list)

        with nvtx.annotate("Feature Aggregation", color="blue"):
            self.BAM_FS.SA_read_feature_dist(gathered_torch_ptr_list, index_ptr_list, gathered_index_size_list, self.world_size, dim, self.cache_dim, 0, staic_info_ten_ptr_list)
    

        self.agg_time += time.time() - feature_read_start

        # Gathering Minibatch Tensors
        torch.distributed.barrier()

        self.gather_tensors(orig_tensor_list, gathered_tensor_list)


        # GPU to GPU memcpy
        with nvtx.annotate("GPU to GPU Memcpy", color="green"):
            return_torch = {}
            gather_start = time.time()
            return_torch_ptr_list = []
            for k , v in batch[0].items():
                if(len(v) == 0):
                        empty_t = torch.empty((0, dim)).to(self.gids_device)
                        return_torch[k] = empty_t
                else:
                    key_off = 0
                    if (k in self.hetero_map):
                            key_off = self.hetero_map[k]
                    else:
                        print("Cannot find key: ", k, " in the heterograph map!")
                    
                    return_ten =  torch.zeros([len(v),dim], dtype=torch.float, device=self.gids_device)
                    return_torch[k] = return_ten
                    return_torch_ptr_list.append(return_ten.data_ptr())



            dist_gather_ptr_list = self.create_ptr_list(orig_tensor_list)
            meta_ptr_list = self.create_ptr_list(meta_data_list)
            return_torch_ptr_ten = torch.tensor(return_torch_ptr_list, dtype=torch.int64, device=self.gids_device).contiguous()     

            self.BAM_FS.gather_feature_list_hetero(return_torch_ptr_ten.data_ptr(), dist_gather_ptr_list, orig_index_size_list, self.world_size, dim,  self.rank, meta_ptr_list)

            self.Gather_time += (time.time()) - gather_start

            reset_start = time.time()
            self.BAM_FS.reset_node_counter(self.cpu_index_len_tensor_ptr_list, self.world_size)
            self.Reset_time += (time.time()) - reset_start

            self.GIDS_time += time.time() - GIDS_time_start        
        batch2 = (*batch, return_torch) 


        with nvtx.annotate("Prefetching", color="red"):
            self.BAM_FS.prefetch_from_victim_queue()

        # Updating Reuse Value
        wb_index_list_tensor = self.create_ptr_list_tensor_2D(self.wb_gathered_index_list, device, 1)
        wb_size_list_tensor = self.create_tensor_from_list_2D(self.wb_gathered_index_size_list, 1)

        with nvtx.annotate("Update Counters", color="red"):
            self.BAM_FS.update_reuse_counters(wb_index_list_tensor.data_ptr(), wb_size_list_tensor.data_ptr(), self.max_sample_size, self.world_size, self.wb_size)

 

        return batch2



    def dist_SA_fetch_feature_fast_gather(self, dim, it, device):
        GIDS_time_start = time.time()

        self.distribute_index(dim, it, device)

        batch = self.wb_batch_buffer.pop(0)
        orig_index_size_list = self.wb_orig_index_size_list.pop(0)
        gathered_index_size_list = self.wb_gathered_index_size_list.pop(0)
        gathered_index_list = self.wb_gathered_index_list.pop(0)
        meta_data_list = self.wb_meta_data_list.pop(0)
        index_size = len(batch[0]) 

        # print("GPU: ",self.rank, "batch: ", batch)
        #batch_SI = self.get_static_info(batch)
        # print("GPU: ", self.rank," batch SI: ", batch_SI)

        # Feature Aggregation
        orig_tensor_list = []
        gathered_tensor_list = []
        self.alloc_tensors(dim, orig_index_size_list, orig_tensor_list, gathered_index_size_list, gathered_tensor_list)

        feature_read_start = time.time()
        index_ptr_list = self.create_ptr_list(gathered_index_list)
        gathered_torch_ptr_list = self.create_ptr_list(gathered_tensor_list)
        
        torch.cuda.synchronize()
        
        return_torch2 =  torch.zeros([index_size,dim], dtype=torch.float, device=self.gids_device).contiguous()


        static_info_ten_list = []
        staic_info_ten_ptr_list = []
        if(self.eviction_policy == 1 or self.eviction_policy == 3):
            static_info_ten_list, staic_info_ten_ptr_list = self.get_static_info_dist(index_ptr_list, gathered_index_size_list)

        with nvtx.annotate("Feature Aggregation", color="blue"):
            self.BAM_FS.SA_read_feature_dist(gathered_torch_ptr_list, index_ptr_list, gathered_index_size_list, self.world_size, dim, self.cache_dim, 0, staic_info_ten_ptr_list)
    

        self.agg_time += time.time() - feature_read_start

        # Gathering Minibatch Tensors
        self.gather_tensors(orig_tensor_list, gathered_tensor_list)


        # GPU to GPU memcpy
        with nvtx.annotate("GPU to GPU Memcpy", color="green"):

            gather_start = time.time()
            return_torch =  torch.zeros([index_size,dim], dtype=torch.float, device=self.gids_device).contiguous()

            dist_gather_ptr_list = self.create_ptr_list(orig_tensor_list)
            meta_ptr_list = self.create_ptr_list(meta_data_list)
            self.BAM_FS.gather_feature_list(return_torch.data_ptr(),dist_gather_ptr_list, orig_index_size_list, self.world_size, dim,  self.rank, meta_ptr_list)
            self.Gather_time += (time.time()) - gather_start

            reset_start = time.time()
            self.BAM_FS.reset_node_counter(self.cpu_index_len_tensor_ptr_list, self.world_size)
            self.Reset_time += (time.time()) - reset_start

            self.GIDS_time += time.time() - GIDS_time_start        
        batch2 = (*batch, return_torch) 

        return batch2

    def get_static_info(self, batch):
        index_ten = batch[0]
        index_size = len(index_ten)
        static_info_ten =  torch.zeros(index_size, dtype=torch.uint8, device=self.gids_device).contiguous()

        self.BAM_FS.get_static_info(static_info_ten.data_ptr(), index_ten.data_ptr(), len(index_ten))
        return static_info_ten

    
    def get_static_info_dist(self, index_ptr_list, gathered_index_size_list):

        static_info_ten_list = []
        static_info_ten_ptr_list = []
        for i in range(self.world_size):
            batch_len = gathered_index_size_list[i]
            static_info_ten =  torch.zeros(batch_len, dtype=torch.uint8, device=self.gids_device).contiguous()
            static_info_ten_list.append(static_info_ten)
            static_info_ten_ptr_list.append(static_info_ten.data_ptr())


        self.BAM_FS.get_static_info_dist(static_info_ten_ptr_list, index_ptr_list, gathered_index_size_list)
        return (static_info_ten_list, static_info_ten_ptr_list)



    def print_stats(self):
        print("GIDS time: ", self.GIDS_time)
        wbtime = self.WB_time 
        print("WB time: ", wbtime)
        self.WB_time = 0.0
        self.GIDS_time = 0.0

        print("Sampling Time: ", self.Sampling_time)
        self.Sampling_time = 0.0
        print("Split Time: ", self.Split_time)
        self.Split_time = 0.0
        print("Feat aggregation Time: ", self.agg_time)
        self.agg_time = 0.0
        print("Communication Time: ", self.Communication_time)
        self.Communication_time = 0.0
        print("Communication Setup Time: ", self.Communication_setup_time)
        self.Communication_setup_time = 0.0
        print("Gather Time: ", self.Gather_time)
        self.Gather_time = 0.0
        print("Reset Time: ", self.Reset_time)
        self.Reset_time = 0.0


        self.BAM_FS.print_stats_rank(self.rank)
        
        if (self.graph_GIDS != None):
            self.graph_GIDS.print_stats_no_ctrl()
        return

    # Utility FUnctions
    def store_tensor(self, in_ten, offset):
        num_e = len(in_ten)
        self.BAM_FS.store_tensor(in_ten.data_ptr(),num_e,offset);

    def read_tensor(self, num, offset):
        self.BAM_FS.read_tensor(num, offset)

    def flush_cache(self):
        self.BAM_FS.flush_cache()


