

template <typename T = float>
__global__ void Emulate_SA_read_feature_kernel(Emulate_SA_cache_d_t<T> *cache, T *out_tensor_ptr,
                                    int64_t *index_ptr, int dim,
                                    int64_t num_idx, int cache_dim, uint64_t key_off,  uint8_t* static_info_ptr) {

  uint64_t bid = blockIdx.x;
  int num_warps = blockDim.x / 32;
  int warp_id = threadIdx.x / 32;
  int idx_idx = bid * num_warps + warp_id;
  if (idx_idx < num_idx) {
    uint64_t row_index = index_ptr[idx_idx] + key_off;
    uint64_t tid = threadIdx.x % 32;
   // cache->get_data(0, out_tensor_ptr, idx_idx, static_info_ptr);

    cache->get_data(row_index, out_tensor_ptr + (bid * num_warps + warp_id) * dim, idx_idx, static_info_ptr);
  } 
}


template <typename T = float>
__global__ void Emulate_SA_read_feature_with_color_kernel(Emulate_SA_cache_d_t<T> *cache, T *out_tensor_ptr,
                                    int64_t *index_ptr, int dim,
                                    int64_t num_idx, int cache_dim, uint64_t key_off,  uint8_t* static_info_ptr, 
                                    uint64_t* color_data) {

  uint64_t bid = blockIdx.x;
  int num_warps = blockDim.x / 32;
  int warp_id = threadIdx.x / 32;
  int idx_idx = bid * num_warps + warp_id;
  if (idx_idx < num_idx) {
    uint64_t row_index = index_ptr[idx_idx] + key_off;
    uint64_t tid = threadIdx.x % 32;
   // cache->get_data(0, out_tensor_ptr, idx_idx, static_info_ptr);

    cache->get_data(row_index, out_tensor_ptr + (bid * num_warps + warp_id) * dim, idx_idx, static_info_ptr, color_data[idx_idx]);
  } 
}



template <typename T = float>
__global__ void Emulate_SA_print_counters(Emulate_SA_cache_d_t<T> *cache){
  if(threadIdx.x == 0 && blockIdx.x == 0)
    cache -> print_stats();

  __syncthreads();

}

__global__ void distribute_node_kernel(int64_t* items, float* score_ptr, int64_t* index_ptr, int* counter_ptr, 
                                       float* dict, int64_t* color_p, int item_len, const int num_nvlink , uint64_t num_colors){

  //float scores[num_nvlink];
  extern __shared__ float scores[];

  

  int max_idx = -1;
  float max_val = -1.0f;
  uint64_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  uint64_t off = threadIdx.x * num_nvlink;

  for (int i = 0; i < num_nvlink; i++){
   scores[i+off] = 0.0f; 
  }

  int sec_len = (item_len / num_nvlink);
  if(tid < item_len){
    for (int i = 0 ; i < num_nvlink; i++){
      float cur_score = score_ptr[tid * num_nvlink + i];

      scores[i+off]  = cur_score;
      if(cur_score > max_val) {
        max_idx = i;
        max_val = cur_score;
      }
    }

    
    int insert_idx = atomicAdd(counter_ptr + max_idx, 1);
    //printf("insert idx:%i max_idx:%i sec len: %i\n", insert_idx, max_idx,  sec_len);

    if(insert_idx < sec_len){
      index_ptr[insert_idx * num_nvlink + max_idx] = items[tid];
      auto color = color_p[tid];
      if(max_idx == -1) printf("error\n");
      if(color == -1) printf("error2\n");

      if(color != 0){
        uint64_t d_idx =  max_idx * (num_colors + 1) + color;
        //printf("idx: %llu\n", d_idx);
        atomicAdd(&(dict[d_idx]), 1.0f);
      }
    }
    else{

      scores[max_idx+off] = -0.1f;
      bool print_check = true;
      bool done = false;
      while(!done){
        int max_idx = -1;
        float max_val = -1.0f;
        for (int i = 0 ; i < num_nvlink; i++){
          float cur_score = scores[i+off];
           
          if(cur_score > max_val) {
            max_idx = i;
            max_val = cur_score;
          }
        }

        insert_idx = atomicAdd(counter_ptr + max_idx, 1);

        // if(print_check){
        //   printf("insert idx:%i max_idx:%i sec len: %i\n", insert_idx, max_idx,  sec_len);
        //   print_check = false;
        // }

        // if(max_idx == -1 && print_check){
        //   printf("Test %f %f %f %f\n", scores[off], scores[off+1], scores[off+2], scores[off+3]);
        //   print_check = false;
        // }

        if(insert_idx < sec_len){
          if(max_idx == -1) printf("error\n");

          index_ptr[insert_idx * num_nvlink + max_idx] = items[tid];
          done = true;
          auto color = color_p[tid];
          if(color == -1) printf("error2\n");
          if(color != 0){
            //atomicAdd(&(dict[max_idx * (num_colors + 1) + color]), 1.0);
            uint64_t d_idx =  max_idx * (num_colors + 1) + color;
            //printf("idx: %llu\n", d_idx);
            
            atomicAdd(&(dict[d_idx]), 1.0f);
          }
        }
        else{
          scores[max_idx+off] = -0.1f;

        }
      }
    }


  }

}
