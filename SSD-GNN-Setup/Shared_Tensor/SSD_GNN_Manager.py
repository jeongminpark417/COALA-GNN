from SSD_GNN_Pybind import NVSHMEM_Manager, SSD_GNN_SSD_Controllers, SSD_GNN_NVSHMEM_Cache
from mpi4py import MPI
import cupy as cp
import torch

class NVShmem_Tensor_Manager(object):
    def __init__(self, nvshmem_batch_ptr: int, batch_nbytes: int, nvshmem_index_ptr: int, 
                index_nbytes: int, shape: tuple, device: str):
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

        


class SSD_GNN_Manager(object):
    def __init__(self,
        num_ssds,
        num_elems, 
        ssd_read_offset,
        cache_size,  # Cache Size in MB

        batch_size,
        fan_out,
        dim,
        
        MPI_comm_manager,
        device,
        cache_backend = "nvshmem",

        is_simulation = False,
        ):
        self.device = device
        self.cache_backend = cache_backend
        self.is_simulation = is_simulation
        self.MPI_comm_manager = MPI_comm_manager

        device_id = MPI_comm_manager.local_rank
        self.SSD_Controllers = SSD_GNN_SSD_Controllers(num_ssds, num_elems, ssd_read_offset, device_id, dim, is_simulation)

        self.max_sample_size = batch_size
        for i in fan_out:
            self.max_sample_size *= (int(i)+1)

        if(self.cache_backend == "nvshmem"):
            local_comm_ptr = MPI._addressof(self.MPI_comm_manager.local_comm)
            self.nvshmem_manager = NVSHMEM_Manager(local_comm_ptr, self.MPI_comm_manager.local_rank)

            # nbytes = int(self.max_sample_size * 4 * dim)
            # nvshmem_ptr = self.nvshmem_manager.allocate(nbytes) 
            # print("NVSHMEM Manager allocate 1 done")

            # index_nbytes = int(self.max_sample_size  * 8 * 2)
            # nvshmem_index_ptr = self.nvshmem_manager.allocate(nbytes) 
            # index_shape = [self.MPI_comm_manager.local_size, int(self.max_sample_size * 2)]
            # print("NVSHMEM Manager allocate 2 done")

            # self.NVshmem_tensor_manager = NVShmem_Tensor_Manager(nvshmem_ptr, nbytes, nvshmem_index_ptr, index_nbytes, index_shape, self.device)
            # print("NVSHMEM Tensor mangaer done")

            # #Initializing SSD_GNN Cache
            # self.SSD_GNN_Cache = SSD_GNN_NVSHMEM_Cache(self.SSD_Controllers,  self.MPI_comm_manager.local_size, cache_size)
            # print("NVSHMEM Cache mangaer done")

        elif (self.cache_backend == "nccl"):
            print(f"Unsupported cache backend: {self.cache_backend}")
        else:
            print(f"Unsupported cache backend: {self.cache_backend}")
            return
        
    # def __del__(self):
    #     if(self.cache_backend == "nvshmem"):
    #         self.nvshmem_manager.free(NVshmem_tensor_manager.nvshmem_batch_ptr)
    #         self.nvshmem_manager.free(NVshmem_tensor_manager.nvshmem_index_ptr)
    #         self.nvshmem_manager.finalize()


