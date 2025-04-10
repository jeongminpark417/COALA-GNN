from SSD_GNN_Pybind import SharedUVAManager
import cupy as cp
import torch
from mpi4py import MPI
import ctypes
import torch
import torch.distributed as dist
import numpy as np
from torch.utils.data import DataLoader, Dataset

import time

class NumpyDataset(Dataset):
    def __init__(self, np_array):
        self.data = np_array

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class MPI_Comm_Manager(object):
    def __init__ (self, node = 0):
        self.global_comm = MPI.COMM_WORLD
        self.global_rank = self.global_comm.Get_rank()
        self.global_size = self.global_comm.Get_size()
        
        self.node_id = int(node)
        self.local_comm = self.global_comm.Split(color=self.node_id, key=self.global_rank)
        self.local_rank = self.local_comm.Get_rank()
        self.local_size = self.local_comm.Get_size()
        self.local_comm.Barrier()
        self.dist_local_rank_list = self.local_comm.allgather(self.global_rank)
        self.is_master = self.local_rank == 0

        print(f"Rank: {self.global_rank} Global Size: {self.global_size}")

        #Gather node ids for the master process within the node
        send_id = self.global_rank if self.local_rank == 0 else None
        node_ids = self.global_comm.allgather(send_id)
        self.master_process_list = [nid for nid in node_ids if nid is not None]
        self.num_master_process = len(self.master_process_list)
#        print(f"Rank: {self.global_rank} master_process_list: {self.master_process_list}")

        master_ids = (self.local_comm.allgather(send_id))
        master_id_list= [nid for nid in master_ids if nid is not None]
        assert len(master_id_list) == 1, f"Number of master processes with in node is not 1"
        self.master_process_id = master_id_list[0]


    def initialize_nested_process_group(self):
        dist.init_process_group("nccl", world_size=self.global_size, rank=self.global_rank)
        dist.barrier()
        # self.global_gloo = dist.new_group(backend='gloo')
        # dist.barrier(self.global_gloo)
       # print(f"Rank: {self.global_rank} local_gloo scather: {self.dist_local_rank_list}")

        for master in self.master_process_list:
            #print(f"Rank: {self.global_rank} master {master}")
            if(master == self.master_process_id):
            #    print(f"Rank: {self.global_rank} master {master} local_gloo scather: {self.dist_local_rank_list}")
                self.local_gloo_scatter = dist.new_group(ranks=self.dist_local_rank_list, backend='gloo')
            dist.barrier()
            if(master == self.master_process_id):
             #   print(f"Rank: {self.global_rank} local_gloo gather: {self.dist_local_rank_list}")
                self.local_gloo_gather = dist.new_group(ranks=self.dist_local_rank_list, backend='gloo')
            dist.barrier()


       # print(f"Rank: {self.global_rank} mast init start")
        if(self.is_master):
            self.master_gloo_gather = dist.new_group(ranks=self.master_process_list, backend='gloo')
        #print(f"Rank: {self.global_rank} init process group done")

        dist.barrier()

        #   dist.init_process_group("nccl", world_size=self.global_size, rank=self.global_rank)
        # dist.barrier()
        # # self.global_gloo = dist.new_group(backend='gloo')
        # # dist.barrier(self.global_gloo)
        # print(f"Rank: {self.global_rank} local_gloo scather: {self.dist_local_rank_list}")

        # self.local_gloo_scatter = dist.new_group(ranks=self.dist_local_rank_list, backend='gloo')
        # dist.barrier()
        # print(f"Rank: {self.global_rank} local_gloo gather: {self.dist_local_rank_list}")
        # self.local_gloo_gather = dist.new_group(ranks=self.dist_local_rank_list, backend='gloo')
        # dist.barrier()
        # print(f"Rank: {self.global_rank} master_gloo_gather  start gather: {self.master_process_list}")

        # if(self.is_master):
        #     self.master_gloo_gather = dist.new_group(ranks=self.master_process_list, backend='gloo')
        # print(f"Rank: {self.global_rank} init process group done")

        # dist.barrier()

    def gather_cache_meta(self, gpu_cache_meta, gathered_data):
        dist.all_reduce(gpu_cache_meta, op=dist.ReduceOp.SUM, group=self.local_gloo_gather)
        dist.barrier(group=self.local_gloo_gather) 
        if(self.is_master):
            dist.all_gather(gathered_data, gpu_cache_meta, group=self.master_gloo_gather)

    def broadcast_training_nodes(self, parsed_training_node_list):
        dist.broadcast(parsed_training_node_list, src=self.master_process_id, group=self.local_gloo_scatter)

    def destroy_process_group(self):
        dist.destroy_process_group()


class MemoryOwner:
    pass

class Shared_UVA_Tensor_Manager(object):
    def __init__(self, 
                 comm_manager,
                 path,
                 tensor_size: int, # in Bytes
                 ):
        self.comm_manager = comm_manager
        comm = comm_manager.global_comm
        node_id = comm_manager.node_id

        global_comm_ptr = MPI._addressof(self.comm_manager.global_comm)
        local_comm_ptr = MPI._addressof(self.comm_manager.local_comm)

        self.memory_handle = SharedUVAManager(path, tensor_size, node_id, global_comm_ptr, local_comm_ptr)
        self.tensor_size = tensor_size
        self.device_ptr = self.memory_handle.get_device_ptr()
        self.host_ptr = self.memory_handle.get_host_ptr()

        self.device = "cuda:" + str(comm_manager.local_rank)

    
        self.owner = MemoryOwner()

    def get_tensor(self, 
                   dtype,
                   device,
                   tensor_shape
                   ):
#        with cp.cuda.Device(self.comm_manager.local_rank):
        #self.allocated_mem = cp.cuda.UnownedMemory(self.device_ptr, self.tensor_size, device_id = self.comm_manager.local_rank, owner=None)
        self.allocated_mem = cp.cuda.UnownedMemory(self.device_ptr, self.tensor_size, owner=None)

        self.memptr = cp.cuda.MemoryPointer(self.allocated_mem, offset=0)
        # print(f"Mem device_id: {self.memptr.device_id}")
        # print(f"rank: {self.comm_manager.local_rank} shape: {tensor_shape} tensro size: {self.tensor_size}")

        uva_array = cp.ndarray(tensor_shape, dtype=dtype, memptr=self.memptr)
        uva_tensor = torch.as_tensor(uva_array, device=self.device)
        #uva_tensor = torch.as_tensor(uva_array)
        #uva_tensor = torch.as_tensor(uva_array)
       # tensor = _C._from_dlpack(uva_array.toDlpack())  # not public API

        #uva_tensor = torch.utils.dlpack.from_dlpack(uva_array.toDlpack())

        return uva_tensor

    def write_np_array(self, uva_tensor, np_array):
        if(self.comm_manager.local_rank == 0):
            if uva_tensor.shape != np_array.shape:
                raise ValueError(f"Tensor shape {uva_tensor.shape} does not match numpy array shape {np_array.shape}")
            load_start = time.time()
            uva_tensor.copy_(torch.from_numpy(np_array))
            loading_time = (time.time() - load_start)
            print(f"Data loading time: {loading_time}")
        self.comm_manager.local_comm.Barrier()

    def write_np_array_gpu(self, uva_tensor, np_array, device):
        if(self.comm_manager.local_rank == 0):
            if uva_tensor.shape != np_array.shape:
                raise ValueError(f"Tensor shape {uva_tensor.shape} does not match numpy array shape {np_array.shape}")
            load_start = time.time()
            
            dataset = NumpyDataset(np_array)
            dataloader = DataLoader(dataset, batch_size=int(1024*1024*1024), shuffle=False, drop_last=False)
            offset = 0
            for idx, batch in enumerate(dataloader):
                uva_tensor[offset: offset+ batch.size(0)] = batch.to('cuda')
                offset += batch.size(0)

            loading_time = (time.time() - load_start)
            print(f"Data loading time: {loading_time} Num copied elements: {offset}")
        self.comm_manager.local_comm.Barrier()







