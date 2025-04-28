from COALA_GNN_Pybind import NVSHMEM_Manager, SSD_GNN_SSD_Controllers, SSD_GNN_NVSHMEM_Cache, Isolated_Cache
from mpi4py import MPI
import cupy as cp
import torch
import torch.distributed as dist
import time

class NVShmem_Tensor_Manager(object):
    def __init__(self, nvshmem_batch_ptr: int, batch_nbytes: int, nvshmem_index_ptr: int, 
                index_nbytes: int, shape: tuple, device: str):
        #print(f"NVSHMEM batch: {batch_nbytes} index: {index_nbytes}")
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

        


class COALA_GNN_Manager(object):
    def __init__(self,
        node_distributor,
        num_ssds,
        page_size,
        num_elems, 
        ssd_read_offset,
        cache_size,  # Cache Size in MB

        batch_size,
        fan_out,
        dim,
        
        MPI_comm_manager,
        device,
        cache_backend = "nvshmem",

        sim_buf = None
        ):
        self.node_distributor = node_distributor
        self.device = device
        self.cache_backend = cache_backend
        self.MPI_comm_manager = MPI_comm_manager
        self.dim = dim
        self.sim_buf = sim_buf
        if(sim_buf == None):
            self.is_simulation = False
        else:
            self.is_simulation = True

        self.aggregation_timer = 0.0

        device_id = MPI_comm_manager.local_rank
        self.SSD_Controllers = SSD_GNN_SSD_Controllers(num_ssds, page_size, num_elems, ssd_read_offset, device_id, dim, self.is_simulation)
        
        self.max_sample_size = batch_size
        for i in fan_out:
            self.max_sample_size *= (int(i)+1)

        if(self.cache_backend == "nvshmem"):
            local_comm_ptr = MPI._addressof(self.MPI_comm_manager.local_comm)
            self.nvshmem_manager = NVSHMEM_Manager(local_comm_ptr, self.MPI_comm_manager.local_rank)

            nbytes = int(self.max_sample_size * 4 * dim)
            nvshmem_ptr = self.nvshmem_manager.allocate(nbytes) 

            index_nbytes = int(self.max_sample_size  * 8 * 2)
            nvshmem_index_ptr = self.nvshmem_manager.allocate(nbytes) 
            index_shape = [self.MPI_comm_manager.local_size, int(self.max_sample_size * 2)]

            self.NVshmem_tensor_manager = NVShmem_Tensor_Manager(nvshmem_ptr, nbytes, nvshmem_index_ptr, index_nbytes, index_shape, self.device)
            # Initializing COALA_GNN Cache
            if(self.is_simulation):
                self.COALA_GNN_Cache = SSD_GNN_NVSHMEM_Cache(self.SSD_Controllers,  self.node_distributor.distribute_manager, self.MPI_comm_manager.global_rank, self.MPI_comm_manager.local_size, cache_size, self.sim_buf.data_ptr())
            else:
                self.COALA_GNN_Cache = SSD_GNN_NVSHMEM_Cache(self.SSD_Controllers,  self.node_distributor.distribute_manager, self.MPI_comm_manager.global_rank, self.MPI_comm_manager.local_size, cache_size, 0)

        elif (self.cache_backend == "isolated" or self.cache_backend == "nccl"):
            if(self.is_simulation):
                self.COALA_GNN_Cache = Isolated_Cache(self.SSD_Controllers,  self.node_distributor.distribute_manager, self.MPI_comm_manager.global_rank, self.MPI_comm_manager.local_size, cache_size, self.sim_buf.data_ptr())
            else:
                self.COALA_GNN_Cache = Isolated_Cache(self.SSD_Controllers,  self.node_distributor.distribute_manager, self.MPI_comm_manager.global_rank, self.MPI_comm_manager.local_size, cache_size, 0)

            if(self.cache_backend == "nccl"):
                node_torch_size = int(self.max_sample_size * self.MPI_comm_manager.local_size)
                self.nccl_node_tensor = torch.zeros(node_torch_size, dtype=torch.int64, device=self.device).contiguous()
                self.nccl_map_tensor =  torch.zeros(node_torch_size, dtype=torch.int64, device=self.device).contiguous()
                self.nccl_counter_tensor = torch.zeros(int(self.MPI_comm_manager.local_size), dtype=torch.int64, device=self.device).contiguous()

        else:            
            print(f"Unsupported cache backend: {self.cache_backend}")
            return


    def fetch_feature(self, batch):
        index = batch[0].to(self.device)
        index_size = len(index)
        index_ptr = index.data_ptr()
        fetch_start = time.time()
    
        if(self.cache_backend == "nvshmem"):
            return_torch_shape = [index_size, self.dim]
            #print(f"return torch shape: {return_torch_shape}")
            return_torch = self.NVshmem_tensor_manager.get_batch_tensor(return_torch_shape)
            return_torch_ptr = self.NVshmem_tensor_manager.get_batch_tensor_ptr()
            request_tensor_ptr = self.NVshmem_tensor_manager.get_index_tensor_ptr()

            self.COALA_GNN_Cache.send_requests(index_ptr, index_size, request_tensor_ptr, self.max_sample_size)
            self.COALA_GNN_Cache.read_feature(return_torch_ptr, request_tensor_ptr, self.max_sample_size)
            
            self.aggregation_timer += (time.time() - fetch_start)
            return (*batch, return_torch)  

        elif(self.cache_backend == "isolated"):
            return_torch =  torch.zeros([index_size, self.dim], dtype=torch.float, device=self.device)
            self.COALA_GNN_Cache.read_feature(return_torch.data_ptr(), index_ptr, index_size)
            self.aggregation_timer += (time.time() - fetch_start)
            return (*batch, return_torch) 
            
        elif(self.cache_backend == "nccl"):
            #$print("Unsupported cache backend for fetch_feature")
            return_torch_shape = [index_size,self.dim]
            return_torch =  torch.zeros(return_torch_shape, dtype=torch.float, device=self.device).contiguous()

            self.nccl_node_tensor.zero_()
            self.nccl_map_tensor.zero_()
            self.nccl_counter_tensor.zero_()
            torch.cuda.synchronize()


            self.COALA_GNN_Cache.split_node_list(index_ptr, index_size, self.nccl_node_tensor.data_ptr(), self.nccl_map_tensor.data_ptr(), 
                self.nccl_counter_tensor.data_ptr(), self.MPI_comm_manager.local_size, self.max_sample_size)

            size_of_buffer = self.max_sample_size
            N = self.MPI_comm_manager.local_size
            split_tensors = [self.nccl_node_tensor[i * size_of_buffer:(i + 1) * size_of_buffer] for i in range(N)]
            recv_tensors = [torch.zeros(size_of_buffer, dtype=torch.int64, device=self.device).contiguous() for _ in range(N)]
            dist.barrier(self.MPI_comm_manager.nccl_cache_gather)

            dist.all_to_all(recv_tensors, split_tensors, group=self.MPI_comm_manager.nccl_cache_gather)
            gathered_counter = torch.zeros(int(N), dtype=torch.int64, device=self.device).contiguous()
            dist.all_to_all_single(gathered_counter, self.nccl_counter_tensor,  group=self.MPI_comm_manager.nccl_cache_gather)

            gathered_feat_tensor_list = []
            gathered_feat_tensor_ptr_list = []

            orig_feat_tensor_list = []
            orig_feat_tensor_ptr_list = []

            for i in range(N):
                gathered_ten_shape = [gathered_counter[i], self.dim]
                gathered_feat_tensor = torch.zeros(gathered_ten_shape, dtype=torch.float32, device=self.device).contiguous()
                gathered_feat_tensor_list.append(gathered_feat_tensor)
                gathered_feat_tensor_ptr_list.append(gathered_feat_tensor.data_ptr())

                orig_ten_shape = [self.nccl_counter_tensor[i], self.dim]
                orig_feat_tensor = torch.zeros(orig_ten_shape, dtype=torch.float32, device=self.device).contiguous()
                orig_feat_tensor_list.append(orig_feat_tensor)
                orig_feat_tensor_ptr_list.append(orig_feat_tensor.data_ptr())

            cpu_nccl_counter_list = self.nccl_counter_tensor.tolist() 

            recv_ptr_tensors = [tensor.data_ptr() for tensor in recv_tensors]

            self.COALA_GNN_Cache.nccl_get_feature(recv_ptr_tensors, gathered_feat_tensor_ptr_list, cpu_nccl_counter_list,
                                   N, self.max_sample_size)
            
            mastet_node_offset = int(self.MPI_comm_manager.master_process_index * N)


            for i in range(N):
                if i == self.MPI_comm_manager.local_rank:
                    # Skip sending data to itself
                    for j in range(N):
                        if j == i:
                            orig_feat_tensor_list[i].copy_(gathered_feat_tensor_list[i])
                        else:
                            dist.recv(orig_feat_tensor_list[j], src=(mastet_node_offset+j),  group=self.MPI_comm_manager.nccl_cache_gather)
                else:
                    dist.send(gathered_feat_tensor_list[i], dst=(mastet_node_offset+i),  group=self.MPI_comm_manager.nccl_cache_gather)

                    #Receive the corresponding tensor from GPU `i` into `orig_feat_tensor_list[rank]`
                    
            # # REMAP
            self.COALA_GNN_Cache.map_feat_data(return_torch.data_ptr(), orig_feat_tensor_ptr_list, self.nccl_map_tensor.data_ptr(), 
                cpu_nccl_counter_list,  N, self.max_sample_size)
            self.aggregation_timer += (time.time() - fetch_start)
            return (*batch, return_torch)
        else:
            print("Unsupported cache backend for fetch_feature")


    def get_cache_data(self, ptr):
        self.COALA_GNN_Cache.get_cache_data(ptr)
        

    def print_stats(self):
        self.COALA_GNN_Cache.print_stats()

    def get_aggregate_time(self):
        return self.aggregation_timer

    def __del__(self):
        if(self.cache_backend == "nvshmem"):
            self.nvshmem_manager.free(self.NVshmem_tensor_manager.nvshmem_batch_ptr)
            self.nvshmem_manager.free(self.NVshmem_tensor_manager.nvshmem_index_ptr)
            self.nvshmem_manager.finalize()


