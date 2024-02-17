

template <typename T = float>
__global__ void read_feature_kernel(array_d_t<T> *dr, T *out_tensor_ptr,
                                    int64_t *index_ptr, int dim,
                                    int64_t num_idx, int cache_dim, uint64_t key_off) {

  uint64_t bid = blockIdx.x;
  int num_warps = blockDim.x / 32;
  int warp_id = threadIdx.x / 32;
  int idx_idx = bid * num_warps + warp_id;


  if (idx_idx < num_idx) {
 	    bam_ptr<T> ptr(dr);

        uint64_t row_index = index_ptr[idx_idx] + key_off;
      	uint64_t tid = threadIdx.x % 32;


    for (; tid < dim; tid += 32) {
	    T temp = ptr[(row_index) * cache_dim + tid];
	    out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = temp;
    }
  }
}


template <typename T = float>
__global__ void SA_read_feature_kernel(SA_cache_d_t<T> *cache, T *out_tensor_ptr,
                                    int64_t *index_ptr, int dim,
                                    int64_t num_idx, int cache_dim, uint64_t key_off, uint32_t head_ptr, uint8_t* static_info_ptr, uint8_t update_counter) {

  uint64_t bid = blockIdx.x;
  int num_warps = blockDim.x / 32;
  int warp_id = threadIdx.x / 32;
  int idx_idx = bid * num_warps + warp_id;
  if (idx_idx < num_idx) {
    uint64_t row_index = index_ptr[idx_idx] + key_off;
    uint64_t tid = threadIdx.x % 32;

    cache->get_data(row_index, out_tensor_ptr + (bid * num_warps + warp_id) * dim, head_ptr, idx_idx, static_info_ptr, update_counter);
  } 
}

template <typename T = float>
__global__ void SA_read_feature_kernel_with_PVP(SA_cache_d_t<T> *cache, T *out_tensor_ptr,
                                    int64_t *index_ptr, uint64_t** node_flag_ptr, T* PVP_pinned_data,  int dim,
                                    int64_t num_idx, int cache_dim, int cur_gpu, uint64_t key_off, uint32_t head_ptr, uint8_t* static_info_ptr, uint8_t update_counter, bool debug_mode, unsigned long long* debug_count = nullptr) {

  uint64_t bid = blockIdx.x;
  int num_warps = blockDim.x / 32;
  int warp_id = threadIdx.x / 32;
  uint64_t idx_idx = bid * num_warps + warp_id;
  //if(idx_idx == 0 && threadIdx.x == 0) printf("GPU ID:%llu num_idx size: %llu\n", (unsigned long long) (cache -> my_GPU_id_), (unsigned long long) num_idx);

  if (idx_idx < num_idx) {
    uint64_t row_index = index_ptr[idx_idx] + key_off;
    uint64_t tid = threadIdx.x % 32;


    uint64_t fetch_idx = node_flag_ptr[cur_gpu][idx_idx] ;

    //already prefetched
   if((fetch_idx >> 63) == (uint64_t) 1){
      //if(tid == 0) printf("prefetched KEY:%llu my GPU ID:%llu write GPU:%i IDX:%llu num_idx:%llu \n", row_index, (unsigned long long) (cache -> my_GPU_id_), cur_gpu,  (unsigned long long) idx_idx,(unsigned long long) num_idx);
      if(debug_mode) {
        if(tid == 0) atomicAdd(debug_count, 1);

      }
      // for(; tid < dim; tid += 32){
      //   uint64_t prefetched_idx = fetch_idx & (0x7FFFFFFFFFFFFFFF);
      //   out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = PVP_pinned_data[dim * fetch_idx + tid];
      // }
      T* PVP_data_ptr = PVP_pinned_data + dim * fetch_idx;
      cache->get_data_from_PVP(row_index, out_tensor_ptr + (bid * num_warps + warp_id) * dim, head_ptr,idx_idx, static_info_ptr, update_counter, PVP_data_ptr);

    }

    else{
      cache->get_data(row_index, out_tensor_ptr + (bid * num_warps + warp_id) * dim, head_ptr,idx_idx, static_info_ptr, update_counter);
    }
  }
  
}



template <typename T = float>
__global__ void read_feature_kernel_with_cpu_backing_memory(array_d_t<T> *dr, range_d_t<T> *range, T *out_tensor_ptr,
                                    int64_t *index_ptr, int dim,
                                    int64_t num_idx, int cache_dim, GIDS_CPU_buffer<T> CPU_buffer, bool cpu_seq, unsigned int* d_cpu_access, uint64_t key_off) {

  uint64_t bid = blockIdx.x;

  int num_warps = blockDim.x / 32;
  int warp_id = threadIdx.x / 32;
  int idx_idx = bid * num_warps + warp_id;
  if (idx_idx < num_idx) {
 	    bam_ptr<T> ptr(dr);

      uint64_t row_index = index_ptr[idx_idx] + key_off;
      uint64_t tid = threadIdx.x % 32;

      uint32_t cpu_off = range -> get_cpu_offset(row_index);


      if(cpu_seq){
        if(row_index < CPU_buffer.cpu_buffer_len){
          if(tid == 0)
            atomicAdd(d_cpu_access, 1);
          for (; tid < dim; tid += 32) {
            T temp = CPU_buffer.device_cpu_buffer[(row_index) * cache_dim + tid];
            out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = temp;
            }
        }

        else{
        for (; tid < dim; tid += 32) {
          T temp = ptr[(row_index) * cache_dim + tid];
          out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = temp;
        }
      }
      }
      else{
        if((cpu_off & 0x1) == 1){
          if(tid == 0)
            atomicAdd(d_cpu_access, 1);

            for (; tid < dim; tid += 32) {
              T temp = CPU_buffer.device_cpu_buffer[(cpu_off >> 1) * cache_dim + tid];
              out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = temp;
            }
        }

        else{
          for (; tid < dim; tid += 32) {
            T temp = ptr[(row_index) * cache_dim + tid];
            out_tensor_ptr[(bid * num_warps + warp_id) * dim + tid] = temp;
          }
        }
      }
  }
}


template <typename T = float>
__global__ void set_cpu_buffer_kernel(range_d_t<T> *d_range, uint64_t* idx_ptr, int num, uint32_t pageSize) {
  
  uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if(idx <  num){
    d_range -> set_cpu_buffer(idx_ptr[idx], idx );
  }
}


template <typename T = float>
__global__
void set_window_buffering_kernel(array_d_t<T>* dr, uint64_t *index_ptr, uint64_t page_size, int hash_off){
	bam_ptr<T> ptr(dr);
	if(threadIdx.x == 0){
		uint64_t page_idx = index_ptr[blockIdx.x] + hash_off;
		ptr.set_window_buffer_counter(page_idx * page_size/sizeof(T), 1);
	}
}

template <typename T = float>
__global__ void read_kernel(array_d_t<T> *dr,
                                    uint64_t num, uint64_t offset) {
      bam_ptr<T> ptr(dr);
     if(threadIdx.x == 0 && blockIdx.x == 0){
        for(uint64_t i = 0; i < num; i++){
              if(i == 0) printf("idx: %llu type size:%i \n", offset,  (int) sizeof(T));
             // T temp = ptr[i + offset];
              printf("read data: %llu\n",  (unsigned long long) ptr[i + offset]);
             // printf("float read data: %f\n", temp);

        }
     }                           
}

template <typename T = float>
__global__ void write_feature_kernel(Controller** ctrls, page_cache_d_t* pc, array_d_t<T> *dr, T* in_tensor_ptr,
                                    uint64_t num, uint64_t offset) {

    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < num){
      bam_ptr<T> ptr(dr);
      ptr[idx + offset] = in_tensor_ptr[idx];
    }
}


template <typename T = float>
__global__ void 
split_node_list_init_kernel(int64_t* index_ptr, uint64_t* index_pointer_list,  int64_t num_gpu,  int64_t index_size){
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < index_size){
    int64_t cur_node = index_ptr[idx];
    int64_t gpu_id = cur_node % num_gpu;
    uint64_t counter_add = (index_pointer_list[gpu_id]);
    atomicAdd((unsigned int*) (index_pointer_list[gpu_id]), (unsigned int )1);
  }
}

template <typename T = float>
__global__ void 
split_node_list_kernel(int64_t* index_ptr, uint64_t* dist_index_ptr,  uint64_t* index_pointer_list,  int64_t num_gpu, int64_t index_size, 
    uint64_t* meta_buffer_ptr){
  
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx < index_size){
    int64_t cur_node = index_ptr[idx];
    int64_t gpu_id = cur_node % num_gpu;
    unsigned long long int enq_idx = atomicAdd((unsigned long long int*) (index_pointer_list[gpu_id]), (unsigned long long int)1);

    int64_t* dist_index = (int64_t*) (dist_index_ptr[gpu_id]);
    uint64_t* meta_buffer = (uint64_t*) (meta_buffer_ptr[gpu_id]);
    meta_buffer[enq_idx] = idx;
    dist_index[enq_idx] = cur_node;

  }

}


template <typename T = float>
 __forceinline__
__device__
void block_memcpy(void* dst, void* src, size_t size){
     T* src_ptr = (T*) src;
     T* dst_ptr = (T*) dst;
     
     uint32_t count = blockDim.x;     
     uint32_t my_id = threadIdx.x;

     for(; my_id < size; my_id += count){
          dst_ptr[my_id] =  src_ptr[my_id]; 
     }
 }



template <typename T = float>
__global__ 
void 
gather_feature_kernel(T *out_tensor_ptr, T* src_tensor_ptr, uint64_t* meta_buffer, int dim, int64_t num_idx, int rank, int my_rank){

  uint64_t r_idx = blockIdx.x;
  if(r_idx < num_idx){
    uint64_t dst_idx = meta_buffer[r_idx];
    // if(dst_idx == 1 && threadIdx.x == 0) {
    //   printf("my rank: %i src rank:%i r_idx:%llu src data 1: %f 2: %f\n", rank, my_rank, (unsigned long long) r_idx, (float) ((src_tensor_ptr + r_idx * dim)[0]),  (float) ((src_tensor_ptr + r_idx * dim)[1]));
    // }
    
    //block_memcpy<T>((void*) (out_tensor_ptr + dst_idx * dim), (void*)(src_tensor_ptr + r_idx * dim), dim * sizeof(T) / sizeof(T) );
    block_memcpy<uint64_t>((void*) (out_tensor_ptr + dst_idx * dim), (void*)(src_tensor_ptr + r_idx * dim), dim * sizeof(uint64_t) / sizeof(uint64_t) );

  }


}

template <typename T = float>
__global__ 
void
print_meta_buffer_kernel( uint64_t** d_meta_buffer, uint64_t gpu_id, uint64_t meta_len, uint64_t rank){
    uint64_t* meta_buffer = d_meta_buffer[gpu_id];
    for(int i = 0; i < meta_len; i++){
      printf("rank: %llu meta idx: %llu\n", (unsigned long long) rank, meta_buffer[i]);
    }

}




// Preemptive Victim-buffer Prefetcher
template <typename T = float>
__global__ 
void
update_reuse_counters_kernel(SA_cache_d_t<T> *cache, uint64_t** batch_arrays, uint64_t* batch_size_array, uint32_t num_gpus){
  uint64_t bid = blockIdx.x;
  int num_warps = blockDim.x / 32;
  int warp_id = threadIdx.x / 32;
  uint64_t read_idx = bid * num_warps + warp_id;

  uint64_t y_bid = blockIdx.y;


  uint32_t reuse_time = (blockIdx.y / num_gpus);
  uint32_t GPU_id = blockIdx.y % num_gpus;
  const uint64_t num_idx = batch_size_array[y_bid];

 // if(bid == 0 && threadIdx.x ==0 && GPU_id != 0) printf("GPU id diff:%lu num_idx: %llu\n", (unsigned long)GPU_id, num_idx);
  //if(bid == 0 && threadIdx.x == 0) printf("reuse time:%i num_idx:%llu\n", (int) reuse_time, (unsigned long long) num_idx);

  if(read_idx < num_idx){
    uint64_t* index_ptr =(uint64_t*) (batch_arrays[y_bid]);
    uint64_t node_id = index_ptr[read_idx];
    //if(GPU_id != 0) printf("\t \t GPU_id correct\n");
    
    cache->update_reuse_val(node_id, reuse_time, GPU_id, read_idx,num_idx);
  }
  
}




template <typename T = float>
__global__ 
void fill_batch_kernel(uint64_t* PVP_pinned_idx, uint64_t** node_flag_ptr, uint32_t batch_size, int dim, bool debug_mode, unsigned  long long* debug_counter, unsigned int my_GPU, uint64_t max_sample_size) {


  uint64_t id = blockIdx.x * blockDim.x + threadIdx.x;
//  if(id == 0) printf("GPU: %llu batch size; %llu sizeof data:%llu\n",(unsigned long long)my_GPU, (unsigned long long) batch_size,  (unsigned long long)(sizeof(unsigned long long)));

  if (id < batch_size){
    uint64_t cur_idx =  PVP_pinned_idx[id];
    uint16_t cur_GPU_ID = (cur_idx >> 40) & 0x00FF;
    uint64_t batch_idx = (cur_idx & (0x000000FFFFFFFFFF));
    //printf("node ID Write id: %llu idx:%llu my GPU ID: %llu write GPU_id: %llu\n", id, batch_idx, (unsigned long long) my_GPU, (unsigned long long) cur_GPU_ID);
    uint64_t flag = id | 0x8000000000000000;

    if(batch_idx >=  max_sample_size || cur_GPU_ID > 1){
      printf("out of index GPU:%llu index: %llu GPU id: %llu id:%llu\n", (unsigned long long)my_GPU, (unsigned long long) batch_idx, (unsigned long long) cur_GPU_ID, (unsigned long long) id);
    }
      //printf("FILL GPU:%llu  index: %llu GPU id: %llu\n",(unsigned long long)my_GPU, (unsigned long long) batch_idx, (unsigned long long) cur_GPU_ID);

    node_flag_ptr[cur_GPU_ID][batch_idx] = (id | 0x8000000000000000);

    if(id == 0 && debug_mode) {
      atomicAdd(debug_counter, batch_size);
      //printf("batch_size: %llu\n", (unsigned long long) batch_size);
    }
    
  }

  else{

  }

}



__global__
void
get_static_info_kernel(uint8_t* out_ptr, uint64_t* index_ptr, uint64_t index_len, uint8_t* static_val_array ){
  uint64_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id < index_len){
    uint64_t node_id  = index_ptr[id];
    uint8_t static_val = static_val_array[node_id];
    out_ptr[id] = static_val;
  }

}


template <typename T = float>
__global__ 
void
print_kernel(SA_cache_d_t<T> *cache, bool debug_mode, unsigned long long*  evict_counter, unsigned long long* prefetch_counter){
  cache -> print_stats();
  if(threadIdx.x == 0 && debug_mode) {
    printf("evict count: %llu\n", evict_counter[0]);
    printf("prefetch count: %llu\n", prefetch_counter[0]);

  }
}
