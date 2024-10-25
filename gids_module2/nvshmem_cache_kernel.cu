template <typename T = float>
__global__ void NVShmem_read_feature_kernel(NVSHMEM_cache_d_t<T> *cache, T *out_tensor_ptr,
                                    int64_t *index_ptr, int dim,
                                    int64_t num_idx, int cache_dim) {

  uint64_t bid = blockIdx.x;
  int num_warps = blockDim.x / 32;
  int warp_id = threadIdx.x / 32;
  int idx_idx = bid * num_warps + warp_id;
  if (idx_idx < num_idx) {
    uint64_t row_index = index_ptr[idx_idx];
    uint64_t tid = threadIdx.x % 32;

    //cache->get_data(row_index, out_tensor_ptr + (bid * num_warps + warp_id) * dim, idx_idx, nullptr);
    cache->get_data(row_index, out_tensor_ptr + (bid * num_warps + warp_id) * dim, false, 0, 0);

  } 
}

template <typename T = float>
__global__ void NVShmem_dist_read_feature_kernel(int gpu_id, NVSHMEM_cache_d_t<T> *cache, T *out_tensor_ptr,
                                    int64_t *nvshmem_index_ptr, int dim,
                                    int64_t num_idx, int cache_dim, int rank) {

  uint64_t bid = blockIdx.x;
  int num_warps = blockDim.x / 32;
  int warp_id = threadIdx.x / 32;
  int idx_idx = bid * num_warps + warp_id;
  if (idx_idx < num_idx) {
    //Request is a pair (node id, batch idx)
    int64_t row_index = nvshmem_index_ptr[idx_idx*2];
    int64_t batch_idx = nvshmem_index_ptr[idx_idx*2 + 1];
    uint64_t tid = threadIdx.x % 32;


    //get_data( uint64_t id, T* output_ptr, bool use_nvshmem = false, int rank = 0, int dst_gpu = 0){

    //cache->dist_get_data(row_index, out_tensor_ptr + (batch_idx) * dim, rank, gpu_id);
    cache->get_data(row_index, out_tensor_ptr + (batch_idx) * dim, true, rank, gpu_id);

  } 
}



template <typename T = float>
__global__ void send_requests_kernel(int64_t *src_index_ptr, int64_t num_idx,int64_t * nvshmem_request_ptr, 
                                  int rank, int nranks, unsigned int* counters) {

  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if(tid < num_idx){
    int64_t node_id = src_index_ptr[tid];
    int dest_pe_id = node_id % nranks;
    
    unsigned int dest_idx = atomicAdd(counters + dest_pe_id, 1) * 2;
    //unsigned int dest_idx = atomicAdd(counters, 1);

    // if(rank == 0 && dest_pe_id == 1){
    //   printf("Send node id: %lli dest_idx: %u\n", (unsigned long long) node_id, dest_idx);
    // }


    nvshmem_int64_p(nvshmem_request_ptr + dest_idx, node_id, dest_pe_id);
    nvshmem_int64_p(nvshmem_request_ptr + (dest_idx + 1), tid, dest_pe_id);


  }

  // nvshmem_quiet();
  // nvshmem_barrier_all();

}


// Should Optimize
template <typename T = float>
__global__ void NVShmem_count_requests_kernel(int64_t* nvshmem_index_ptr, unsigned int* request_counters, uint64_t num_idx,  int n_gpus, int rank){
  
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if(tid < num_idx){
    int num_ph = blockDim.y / n_gpus;
    for(int ph = 0; ph < num_ph; ph++){
      
        int ydim = ph * n_gpus + threadIdx.y;
        int off = ydim * num_idx * 2;
        int64_t n_id = 0;
       // n_id =  nvshmem_index_ptr[tid * 2 + off];
          n_id = nvshmem_int64_g(nvshmem_index_ptr + tid*2 + off, rank);
        if(n_id != -1){
          //printf("GPU idx; %i tid: %i ydim: %i n_id: %lli\n", (int) rank,  (int)tid, (int) ydim, (long long int)n_id);
          atomicAdd(request_counters + ydim, 1);
        }

      
    }
  }

}



template <typename T = float>
__global__ 
void
NVShmem_print_kernel(NVSHMEM_cache_d_t<T> *cache){
  cache -> print_stats();
}


template <typename T = float>
__global__ 
void
test_read_kernel(int64_t* ptr, int max_index){

  for(int i = 0; i < max_index * 2; i++){
   // int64_t val = nvshmem_int64_g(ptr + (i * 2), 0);
    int64_t val = ptr[i*2];
    printf("idx:%i val: %lli\n", i, val);
  }
}

