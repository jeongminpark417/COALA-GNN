from COALA_GNN_Pybind import NVSHMEM_Manager, SSD_GNN_SSD_Controllers, SSD_GNN_NVSHMEM_Cache
from mpi4py import MPI
import cupy as cp
import torch

class NVShmem_Tensor_Manager(object):
    def __init__(self, nvshmem_batch_ptr: int, batch_nbytes: int, nvshmem_index_ptr: int, 
                index_nbytes: int, shape: tuple, device: str):
        print(f"NVSHMEM batch: {batch_nbytes} index: {index_nbytes}")
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
            print("NVSHMEM Cache mangaer done")

        elif (self.cache_backend == "nccl"):
            print(f"Unsupported cache backend: {self.cache_backend}")
        else:
            print(f"Unsupported cache backend: {self.cache_backend}")
            return


    def fetch_feature(self, batch):
        index = batch[0].to(self.device)
        index_size = len(index)
        index_ptr = index.data_ptr()

        if(self.cache_backend == "nvshmem"):
            return_torch_shape = [index_size, self.dim]
            #print(f"return torch shape: {return_torch_shape}")
            return_torch = self.NVshmem_tensor_manager.get_batch_tensor(return_torch_shape)
            return_torch_ptr = self.NVshmem_tensor_manager.get_batch_tensor_ptr()
            request_tensor_ptr = self.NVshmem_tensor_manager.get_index_tensor_ptr()

            self.COALA_GNN_Cache.send_requests(index_ptr, index_size, request_tensor_ptr, self.max_sample_size)
            self.COALA_GNN_Cache.read_feature(return_torch_ptr, request_tensor_ptr, self.max_sample_size)

            return (*batch, return_torch)
        else:
            print("Unsupported cache backend for fetch_feature")
            return

    def get_cache_data(self, ptr):
        self.COALA_GNN_Cache.get_cache_data(ptr)
        

    def print_stats(self):
        self.COALA_GNN_Cache.print_stats()

    def __del__(self):
        if(self.cache_backend == "nvshmem"):
            self.nvshmem_manager.free(self.NVshmem_tensor_manager.nvshmem_batch_ptr)
            self.nvshmem_manager.free(self.NVshmem_tensor_manager.nvshmem_index_ptr)
            self.nvshmem_manager.finalize()


