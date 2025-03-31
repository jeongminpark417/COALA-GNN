import time
import torch
import numpy as np
import ctypes
import nvtx 

class Metadata_File_Wrapper():
    def __init__(self, color_file, topk_file, score_file):
        self.color_file = color_file
        self.topk_file = topk_file
        self.score_file = score_file


class SSD_GNN():
    def __init__(self):

        


class SSD_GNN():
    def __init__(self, page_size=4096, off=0, dim = 1024, cache_dim = 1024, num_ele = 300*1000*1000*1024, 
        num_ssd = 1,  
        GPU_cache_size = 10,  
        CPU_cache_size = 0, 

        use_ddp = False, 
        fan_out = None, 
        batch_size = 1024,
        heterogeneous = False, 
        hetero_map=None,
        use_nvshmem_tensor = False,
        nvshmem_test = False,
        is_simulation = False,
        feat_file = "",
        feat_off = 0,
        use_color_data = False,
        color_file = "",
        topk_file = "",
        score_file = "",
        comm_wrapper = None,
        global_world_size = 1,
        parsing_method = "baseline",
        cache_backend = "nvshmem"
        ):

 
        self.cache_backend = cache_backend
        self.global_world_size = global_world_size
        self.use_ddp = use_ddp
        self.use_nvshmem_tensor = use_nvshmem_tensor


        self.comm_wrapper = comm_wrapper
        self.parsing_method = parsing_method
        self.timer = Timer()

        self.is_simulation = is_simulation
        self.feat_file = feat_file
        self.feat_off = feat_off
        

        self.use_color_data = use_color_data
        self.color_file = color_file
        self.topk_file = topk_file
        self.score_file = score_file
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
        self.NVSHMEM_Cache.init(0)
        self.rank = self.NVSHMEM_Cache.get_rank()
        self.world_size = self.NVSHMEM_Cache.get_world_size()
        self.mype_node = self.NVSHMEM_Cache.get_mype_node()

        self.comm_protocol = S3D_Communication_Protocol(self.NVSHMEM_Cache, self.comm_wrapper, batch_size, self.parsing_method)

        if(self.use_nvshmem_tensor or self.nvshmem_test):
           
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


            if(self.nvshmem_test):
                self.cache_per_GPU = 1

            print(f"Rank:{self.rank}  init done")

        else:

            if(cache_backend == "nccl"):
                self.device = "cuda:" + str(self.rank)
                print(f"NCCL backend shared cache, Rank: {self.rank} Cache per GPU: {self.world_size} slurm node id: {self.comm_wrapper[9]} ")
                self.cache_per_GPU = self.world_size
                node_torch_size = int(self.max_sample_size * self.cache_per_GPU)
                self.nccl_node_tensor = torch.zeros(node_torch_size, dtype=torch.int64, device=self.device)
                self.nccl_map_tensor =  torch.zeros(node_torch_size, dtype=torch.int64, device=self.device)
                self.nccl_counter_tensor = torch.zeros(int(self.cache_per_GPU), dtype=torch.int64, device=self.device)


                self.nccl_cache = self.comm_wrapper[10]
                self.slurm_node_id = self.comm_wrapper[9]
                if(self.nccl_cache == None):
                    print("NCCL backend is not set. Set cache_backend to nccl")
                #self.max_sample_size
            else:
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

    def MPI_get_rank(self):
        return  self.NVSHMEM_Cache.MPI_get_rank()
    def MPI_get_world_size(self):
        return  self.NVSHMEM_Cache.MPI_get_world_size()

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
        self.NVSHMEM_Cache.init_cache(self.GIDS_controller, self.page_size, self.off, self.GPU_cache_size, self.CPU_cache_size, self.cache_per_GPU, self.num_ele, self.num_ssd, self.num_ways, self.is_simulation, self.feat_file, self.feat_off,
             self.use_color_data, self.color_file, self.topk_file, self.score_file
        )
    
    
    #Fetching Data from the SSDs
    def fetch_feature(self, dim, it, device):
        if(self.use_nvshmem_tensor):
            batch = self.Dist_Cache_fetch_feature(dim, it, device)
        else:
            if(self.cache_backend == "nccl"):
                batch =  self.NCCL_Cache_fetch_feature(dim, it, device)
            else:
                batch =  self.Independent_Cache_fetch_feature(dim, it, device)
        return batch


    def Dist_Cache_fetch_feature(self, dim, it, device):
        GIDS_time_start = time.time()
        batch = next(it)
        self.timer.sampling_time += (time.time() - GIDS_time_start)
        if(self.heterograph):
            print("HETERO GRAPH IS NOT SUPPORT YET")
            return batch

        else:
            index = batch[0].to(self.gids_device)
            index_size = len(index)
            index_ptr = index.data_ptr()

            
            send_start = time.time()
            return_torch_shape = [index_size,dim]
            return_torch = self.NVshmem_tensor_manager.get_batch_tensor(return_torch_shape)
            return_torch_ptr = self.NVshmem_tensor_manager.get_batch_tensor_ptr()

            request_tensor_ptr = self.NVshmem_tensor_manager.get_index_tensor_ptr()
            self.NVSHMEM_Cache.send_requests(index_ptr, index_size, request_tensor_ptr, self.max_sample_size)
            self.timer.send_request_time += (time.time() - send_start)


            fetch_start = time.time()
            self.NVSHMEM_Cache.dist_read_feature(return_torch_ptr, request_tensor_ptr, self.max_sample_size, dim, self.cache_dim)
            self.timer.fetch_time += (time.time() - fetch_start)

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
        self.timer.sampling_time += (time.time() - GIDS_time_start)
     
        if(self.heterograph):
            print("HETERO GRAPH IS NOT SUPPORT YET")
            return batch

        else:
            index = batch[0].to(self.gids_device)
            #print("indx: ", index)
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

    def NCCL_Cache_fetch_feature(self, dim, it, device):
#        print(f"Rank:{self.rank} NCCL_Cache_fetch_feature")
        GIDS_time_start = time.time()

        batch = next(it)
        self.timer.sampling_time += (time.time() - GIDS_time_start)
     
        if(self.heterograph):
            print("HETERO GRAPH IS NOT SUPPORT YET")
            return batch

        else:
            
            index = batch[0].to(self.gids_device)
            
            index_size = len(index)
            index_ptr = index.data_ptr()

            return_torch_shape = [index_size,dim]
            return_torch =  torch.zeros(return_torch_shape, dtype=torch.float, device=self.gids_device).contiguous()


            # NCCL communication Time
            self.nccl_node_tensor.zero_()
            self.nccl_map_tensor.zero_()
            self.nccl_counter_tensor.zero_()

            self.NVSHMEM_Cache.split_node_list(index_ptr, index_size, self.nccl_node_tensor.data_ptr(), self.nccl_map_tensor.data_ptr(), 
                self.nccl_counter_tensor.data_ptr(), self.cache_per_GPU, self.max_sample_size)

            size_of_buffer = self.max_sample_size
            N = self.cache_per_GPU
            split_tensors = [self.nccl_node_tensor[i * size_of_buffer:(i + 1) * size_of_buffer] for i in range(N)]
            recv_tensors = [torch.zeros(size_of_buffer, dtype=torch.int64, device=self.gids_device) for _ in range(N)]

            dist.all_to_all(recv_tensors, split_tensors, group=self.nccl_cache)

            gathered_counter = torch.zeros(int(self.cache_per_GPU), dtype=torch.int64, device=self.gids_device)
            dist.all_to_all_single(gathered_counter, self.nccl_counter_tensor,  group=self.nccl_cache)

            gathered_feat_tensor_list = []
            gathered_feat_tensor_ptr_list = []

            orig_feat_tensor_list = []
            orig_feat_tensor_ptr_list = []

            for i in range(N):
                gathered_ten_shape = [gathered_counter[i], dim]
                gathered_feat_tensor = torch.zeros(gathered_ten_shape, dtype=torch.float32, device=self.gids_device).contiguous()
                gathered_feat_tensor_list.append(gathered_feat_tensor)
                gathered_feat_tensor_ptr_list.append(gathered_feat_tensor.data_ptr())

                orig_ten_shape = [self.nccl_counter_tensor[i], dim]
                orig_feat_tensor = torch.zeros(orig_ten_shape, dtype=torch.float32, device=self.gids_device).contiguous()
                orig_feat_tensor_list.append(orig_feat_tensor)
                orig_feat_tensor_ptr_list.append(orig_feat_tensor.data_ptr())

            cpu_nccl_counter_list = self.nccl_counter_tensor.tolist() 

            # Fetch feat
            recv_ptr_tensors = [tensor.data_ptr() for tensor in recv_tensors]

            self.NVSHMEM_Cache.read_feature_nccl_backend(recv_ptr_tensors, gathered_feat_tensor_ptr_list, cpu_nccl_counter_list,
                                  dim, self.cache_dim, N)
            
            mastet_node_offset = int(self.slurm_node_id * N)
            #print(f"Rank:{self.rank} master_node_offset: {mastet_node_offset}")
            
            for i in range(N):
                if i == self.rank:
                    # Skip sending data to itself
                    for j in range(N):
                        if j == i:
                            orig_feat_tensor_list[i].copy_(gathered_feat_tensor_list[i])
                        else:
                            dist.recv(orig_feat_tensor_list[j], src=(mastet_node_offset+j),  group=self.nccl_cache)
                else:
                    dist.send(gathered_feat_tensor_list[i], dst=(mastet_node_offset+i),  group=self.nccl_cache)
                    
            # # REMAP
            self.NVSHMEM_Cache.map_feat_data(return_torch.data_ptr(), orig_feat_tensor_ptr_list, self.nccl_map_tensor.data_ptr(), 
                cpu_nccl_counter_list, dim, N)

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