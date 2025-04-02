#include "nvshmem.h"
#include "nvshmemx.h"

__global__ 
void NVSHMEM_send_requests_kernel(int64_t *src_index_ptr, int64_t num_idx,int64_t * nvshmem_request_ptr, 
                                      int num_gpus, unsigned int* counters) {

  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if(tid < num_idx){
    int64_t node_id = src_index_ptr[tid];
    int dest_pe_id = node_id % num_gpus;
    
    unsigned int dest_idx = atomicAdd(counters + dest_pe_id, 1) * 2;
    nvshmem_int64_p(nvshmem_request_ptr + dest_idx, node_id, dest_pe_id);
    nvshmem_int64_p(nvshmem_request_ptr + (dest_idx + 1), tid, dest_pe_id);
  }
}

__global__ 
void NVShmem_count_requests_kernel(int64_t* nvshmem_index_ptr, unsigned int* request_counters, uint64_t num_idx,  int n_gpus, int rank){
  
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if(tid < num_idx){
    int num_ph = blockDim.y / n_gpus;
    for(int ph = 0; ph < num_ph; ph++){
        int ydim = ph * n_gpus + threadIdx.y;
        int off = ydim * num_idx * 2;
        int64_t n_id = 0;
          n_id = nvshmem_int64_g(nvshmem_index_ptr + tid*2 + off, rank);
        if(n_id != -1){
          atomicAdd(request_counters + ydim, 1);
        }
    }
  }
}

template<typename Cache_Type>
__global__ 
void NVShmem_read_feature_kernel(int gpu_id, Cache_Type *cache, float *out_tensor_ptr,
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
    //    get_data(uint64_t id, T* output_ptr, int rank, int dst_gpu){

    cache->get_data(row_index, out_tensor_ptr + (batch_idx) * dim, rank, gpu_id);
  } 
}
