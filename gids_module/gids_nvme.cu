#include <pybind11/pybind11.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>


#include <stdio.h>
#include <vector>

#include <bam_nvme.h>
#include <pybind11/stl.h>
#include "gids_kernel.cu"

#include "set_associative_page_cache.h"
#include "emulate_set_associative_page_cache.h"
//#include <bafs_ptr.h>


typedef std::chrono::high_resolution_clock Clock;

void GIDS_Controllers::init_GIDS_controllers(uint32_t num_ctrls, uint64_t q_depth, uint64_t num_q, 
                          const std::vector<int>& ssd_list, uint32_t device){

  n_ctrls = num_ctrls;
  queueDepth = q_depth;
  numQueues = num_q;
  cudaDevice = device;

  for (size_t i = 0; i < n_ctrls; i++) {
    ctrls.push_back(new Controller(ctrls_paths[ssd_list[i]], nvmNamespace, cudaDevice, queueDepth, numQueues));
  }
}


template <typename TYPE>
void BAM_Feature_Store<TYPE>::cpu_backing_buffer(uint64_t dim, uint64_t len){
  TYPE* cpu_buffer_ptr;
  TYPE* d_cpu_buffer_ptr;

  cuda_err_chk(cudaHostAlloc((TYPE **)&cpu_buffer_ptr, sizeof(TYPE) * dim * len, cudaHostAllocMapped));
  cudaHostGetDevicePointer((TYPE **)&d_cpu_buffer_ptr, (TYPE *)cpu_buffer_ptr, 0);

  CPU_buffer.cpu_buffer_dim = dim;
  CPU_buffer.cpu_buffer_len = len;
  CPU_buffer.cpu_buffer = cpu_buffer_ptr;
  CPU_buffer.device_cpu_buffer = d_cpu_buffer_ptr;
  cpu_buffer_flag = true;
}

template <typename TYPE>
void BAM_Feature_Store<TYPE>::init_controllers(GIDS_Controllers GIDS_ctrl, uint32_t ps, uint64_t read_off, uint64_t cache_size, uint64_t num_ele, uint64_t num_ssd = 1) {

  numElems = num_ele;
  read_offset = read_off;
  n_ctrls = num_ssd;
  this -> pageSize = ps;
  this -> dim = ps / sizeof(TYPE);
  this -> total_access = 0; 

  ctrls = GIDS_ctrl.ctrls;

  cudaDevice = GIDS_ctrl.cudaDevice;


  uint64_t page_size = pageSize;
  uint64_t n_pages = cache_size * 1024LL*1024/page_size;
  this -> numPages = n_pages;

  std::cout << "n pages: " << (int)(this->numPages) <<std::endl;
  std::cout << "page size: " << (int)(this->pageSize) << std::endl;
  std::cout << "num elements: " << this->numElems << std::endl;

  this -> h_pc = new page_cache_t(page_size, n_pages, cudaDevice, ctrls[0][0],(uint64_t)64, ctrls);
  page_cache_t *d_pc = (page_cache_t *)(h_pc->d_pc_ptr);
  uint64_t t_size = numElems * sizeof(TYPE);

  this -> h_range = new range_t<TYPE>((uint64_t)0, (uint64_t)numElems, (uint64_t)read_off,
                              (uint64_t)(t_size / page_size), (uint64_t)0,
                              (uint64_t)page_size, h_pc, cudaDevice, 
			      //REPLICATE
			      STRIPE
			      );

  
  this -> d_range = (range_d_t<TYPE> *)h_range->d_range_ptr;

  this -> vr.push_back(nullptr);
  this -> vr[0] = h_range;
  this -> a = new array_t<TYPE>(numElems, 0, vr, cudaDevice);

  cudaMalloc(&d_cpu_access, sizeof(unsigned int));
  cudaMemset(d_cpu_access, 0 , sizeof(unsigned));
 
  return;
}

template <typename TYPE>
void BAM_Feature_Store<TYPE>::init_set_associative_cache(GIDS_Controllers GIDS_ctrl, uint32_t ps, uint64_t read_off, uint64_t cache_size, uint32_t num_gpus, uint64_t num_ele, uint64_t num_ssd, uint64_t num_ways,
        bool use_WB, bool use_PVP, uint32_t window_buffer_size, uint32_t pvp_depth_size, uint64_t max_sample_size, uint8_t refresh_p, int eviction_policy,  int debug) {

  numElems = num_ele;
  read_offset = read_off;
  n_ctrls = num_ssd;
  n_gpus = num_gpus;
  if(debug == 1)
    debug_mode = true;
  else
    debug_mode = false;

  this -> pageSize = ps;
  this -> dim = ps / sizeof(TYPE);
  this -> total_access = 0; 

  ctrls = GIDS_ctrl.ctrls;
  cudaDevice = GIDS_ctrl.cudaDevice;
  pvp_depth= pvp_depth_size;
  this -> use_PVP = use_PVP;
  this -> max_sample_size = max_sample_size;
  this -> eviction_policy = eviction_policy;

  cudaSetDevice( cudaDevice );

  set_associative_cache = true;
  wb_size = window_buffer_size;

  uint64_t page_size = pageSize;
  uint64_t n_pages = cache_size * 1024LL*1024/page_size;
  this -> numPages = n_pages;

  std::cout << "n pages: " << (int)(this->numPages) <<std::endl;
  std::cout << "page size: " << (int)(this->pageSize) << std::endl;
  std::cout << "num elements: " << this->numElems << std::endl;

  if(debug_mode){
    printf("DEBUG MODE\n");
  }
  else{
    printf("NOT DEBUG MODE\n");
  }

  //uint64_t num_ways = 4;
  cudaDeviceGetStreamPriorityRange(&low_priority, &high_priority);

  uint64_t num_sets = n_pages / num_ways;
  std::cout << "n sets: " << num_sets <<std::endl;
  std::cout << "n ways: " << num_ways << std::endl;
  std::cout << "use PVP: " << use_PVP << std::endl;
  SA_handle = new GIDS_SA_handle<TYPE>(num_sets, num_ways, page_size, ctrls[0][0], ctrls, cudaDevice, use_WB, use_PVP, cudaDevice, num_gpus, eviction_policy, wb_size, pvp_depth_size);

  cache_ptr = SA_handle -> get_ptr();

  if(use_PVP){
    cudaMalloc(&PVP_pinned_data, page_size * pvp_depth_size);
    cudaMalloc(&PVP_pinned_idx, sizeof(uint64_t) * pvp_depth_size);
    node_flag_buffer_array = new uint64_t*[n_gpus];
    for(int i  = 0; i < n_gpus; i++){
      cudaMalloc(&(node_flag_buffer_array[i]), sizeof(uint64_t) * max_sample_size);
      cudaMemset(node_flag_buffer_array[i], 0, sizeof(uint64_t) * max_sample_size);
    }
    cudaMalloc(&node_flag_buffer, sizeof(uint64_t*) * n_gpus);
    cudaMemcpy(node_flag_buffer, node_flag_buffer_array,  sizeof(uint64_t*) * n_gpus, cudaMemcpyHostToDevice);
  }


  cudaMalloc(&d_cpu_access, sizeof(unsigned int));
  cudaMemset(d_cpu_access, 0 , sizeof(unsigned));

 
 
  if(debug_mode){
    cudaMalloc(&evict_counter, sizeof(unsigned long long));
    cudaMalloc(&prefetch_counter, sizeof(unsigned long long));
    cudaMemset(evict_counter, 0, sizeof(unsigned long long));
    cudaMemset(prefetch_counter, 0, sizeof(unsigned long long));
  }

  update_counter = refresh_p - 1;
  refresh_time = refresh_p;
  return;
}


template <typename TYPE>
void  BAM_Feature_Store<TYPE>::create_static_info_buffer(const std::string& path){

  std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << path << std::endl;
        return;
    }

    // Get the size of the file
    size_t dataSize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> buffer(dataSize);

    // Read the content
    if (!file.read(reinterpret_cast<char*>(buffer.data()), dataSize)) {
        std::cerr << "Failed to read file: " << path << std::endl;
        return;
    }

    // Close the file
    file.close();



  cudaHostAlloc((uint8_t **)&h_static_val_array, dataSize , cudaHostAllocMapped);
  std::memcpy(h_static_val_array, buffer.data(), dataSize);
  cudaHostGetDevicePointer((uint8_t **)&d_static_val_array, (uint8_t *)h_static_val_array, 0);
  return;
}




template <typename TYPE>
void  BAM_Feature_Store<TYPE>::set_window_buffering(uint64_t id_idx,  int64_t num_pages, int hash_off = 0){
	 uint64_t* idx_ptr = (uint64_t*) id_idx;
	 uint64_t page_size = pageSize;
	 set_window_buffering_kernel<TYPE><<<num_pages, 32>>>(a->d_array_ptr,idx_ptr, page_size, hash_off);
	 cuda_err_chk(cudaDeviceSynchronize())
}


template <typename TYPE>
void BAM_Feature_Store<TYPE>::print_stats_no_ctrl(){
  std::cout << "print stats: ";
  this->h_pc->print_reset_stats();
  std::cout << std::endl;

  std::cout << "print array reset: ";
  this->a->print_reset_stats();
  std::cout << std::endl;

  
}


template <typename TYPE>
void BAM_Feature_Store<TYPE>::print_stats(){

  if(!set_associative_cache){
    std::cout << "print stats: ";
    this->h_pc->print_reset_stats();
    std::cout << std::endl;

    std::cout << "print array reset: ";
    this->a->print_reset_stats();
    std::cout << std::endl;

  }
  else{
    print_kernel<TYPE><<<1,1>>>(cache_ptr, debug_mode, evict_counter, prefetch_counter);
	  cuda_err_chk(cudaDeviceSynchronize())
  }


    for(int i = 0; i < n_ctrls; i++){
    std::cout << "print ctrl reset " << i << ": ";
      (this->ctrls[i])->print_reset_stats();
      std::cout << std::endl;
    }
  
 
  std::cout << "Kernel Time: \t " << this->kernel_time << std::endl;
  this->kernel_time = 0;
  std::cout << "Total Access: \t " << this->total_access << std::endl;
  this->total_access = 0;

  SA_handle -> print_evicted_cl();

}


template <typename TYPE>
void BAM_Feature_Store<TYPE>::print_stats_rank(uint32_t rank){

  if(!set_associative_cache){
    std::cout << "print stats: ";
    this->h_pc->print_reset_stats();
    std::cout << std::endl;

    std::cout << "print array reset: ";
    this->a->print_reset_stats();
    std::cout << std::endl;

  }
  else{
    print_kernel<TYPE><<<1,1>>>(cache_ptr, debug_mode, evict_counter, prefetch_counter);
	  cuda_err_chk(cudaDeviceSynchronize())
  }


    for(int i = 0; i < n_ctrls; i++){
    std::cout << "GPU: " << rank << " print ctrl reset " << i << ": ";
      (this->ctrls[i])->print_reset_stats();
      std::cout << std::endl;
    }
  
 
  std::cout << "GPU: " << rank << "Kernel Time: \t " << this->kernel_time << std::endl;
  this->kernel_time = 0;
  std::cout << "GPU: " << rank  << "Total Access: \t " << this->total_access << std::endl;
  this->total_access = 0;

  SA_handle -> print_evicted_cl();

}



template <typename TYPE>
void BAM_Feature_Store<TYPE>::read_feature(uint64_t i_ptr, uint64_t i_index_ptr,
                                     int64_t num_index, int dim, int cache_dim, uint64_t key_off = 0) {


  TYPE *tensor_ptr = (TYPE *)i_ptr;
  int64_t *index_ptr = (int64_t *)i_index_ptr;

  uint64_t b_size = blkSize;
  uint64_t n_warp = b_size / 32;
  uint64_t g_size = (num_index+n_warp - 1) / n_warp;

  cuda_err_chk(cudaDeviceSynchronize());
  auto t1 = Clock::now();
  if(cpu_buffer_flag == false){
    read_feature_kernel<TYPE><<<g_size, b_size>>>(a->d_array_ptr, tensor_ptr,
                                                  index_ptr, dim, num_index, cache_dim, key_off);
  }
  else{
    read_feature_kernel_with_cpu_backing_memory<<<g_size, b_size>>>(a->d_array_ptr, d_range, tensor_ptr,
                                                  index_ptr, dim, num_index, cache_dim, CPU_buffer, seq_flag,
                                                  d_cpu_access, key_off);
  }
  cuda_err_chk(cudaDeviceSynchronize());
  cudaMemcpy(&cpu_access_count, d_cpu_access, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  auto t2 = Clock::now();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(
      t2 - t1); // Microsecond (as int)
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      t2 - t1); // Microsecond (as int)
  const float ms_fractional =
      static_cast<float>(us.count()) / 1000; // Milliseconds (as float)

  kernel_time += ms_fractional;
  total_access += num_index;

  return;
}

template <typename TYPE>
void BAM_Feature_Store<TYPE>::read_feature_hetero(int num_iter, const std::vector<uint64_t>&  i_ptr_list, const std::vector<uint64_t>&  i_index_ptr_list,
                                     const std::vector<uint64_t>&   num_index, int dim, int cache_dim, const std::vector<uint64_t>& key_off) {

  cudaStream_t streams[num_iter];
  for (int i = 0; i < num_iter; i++) {
      cudaStreamCreate(&streams[i]);
  }

  cuda_err_chk(cudaDeviceSynchronize());
  auto t1 = Clock::now();

  for(uint64_t i = 0;  i < num_iter; i++){
    uint64_t i_ptr = i_ptr_list[i];
    uint64_t    i_index_ptr =  i_index_ptr_list[i];  
    TYPE *tensor_ptr = (TYPE *) i_ptr;
    int64_t *index_ptr = (int64_t *)i_index_ptr;

    uint64_t b_size = blkSize;
    uint64_t n_warp = b_size / 32;
    uint64_t g_size = (num_index[i]+n_warp - 1) / n_warp;

    if(cpu_buffer_flag == false){
      read_feature_kernel<TYPE><<<g_size, b_size, 0, streams[i] >>>(a->d_array_ptr, tensor_ptr,
                                                    index_ptr, dim, num_index[i], cache_dim, key_off[i]);
    }
    else{
      read_feature_kernel_with_cpu_backing_memory<<<g_size, b_size, 0, streams[i] >>>(a->d_array_ptr, d_range ,tensor_ptr,
                                                    index_ptr, dim, num_index[i], cache_dim, CPU_buffer, seq_flag, 
                                                    d_cpu_access,  key_off[i]);
    }
    total_access += num_index[i];
  }

  for (int i = 0; i < num_iter; i++) {
    cudaStreamSynchronize(streams[i]);
  }

  cuda_err_chk(cudaDeviceSynchronize());
  cuda_err_chk(cudaDeviceSynchronize());
  cudaMemcpy(&cpu_access_count, d_cpu_access, sizeof(unsigned int), cudaMemcpyDeviceToHost);

  auto t2 = Clock::now();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(
      t2 - t1); // Microsecond (as int)
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      t2 - t1); // Microsecond (as int)
  const float ms_fractional =
      static_cast<float>(us.count()) / 1000; // Milliseconds (as float)

  //std::cout << "Duration = " << us.count() << "µs (" << ms_fractional << "ms)"
    //        << std::endl;
 
  kernel_time += ms_fractional;

  for (int i = 0; i < num_iter; i++) {
      cudaStreamDestroy(streams[i]);
  }
  

  return;
}


template <typename TYPE>
void BAM_Feature_Store<TYPE>::read_feature_merged(int num_iter, const std::vector<uint64_t>&  i_ptr_list, const std::vector<uint64_t>&  i_index_ptr_list,
                                     const std::vector<uint64_t>&   num_index, int dim, int cache_dim=1024) {

  cudaStream_t streams[num_iter];
  for (int i = 0; i < num_iter; i++) {
      cudaStreamCreate(&streams[i]);
  }

  cuda_err_chk(cudaDeviceSynchronize());
  auto t1 = Clock::now();

  for(uint64_t i = 0;  i < num_iter; i++){
    uint64_t i_ptr = i_ptr_list[i];
    uint64_t    i_index_ptr =  i_index_ptr_list[i];         
    TYPE *tensor_ptr = (TYPE *) i_ptr;
    int64_t *index_ptr = (int64_t *)i_index_ptr;

    uint64_t b_size = blkSize;
    uint64_t n_warp = b_size / 32;
    uint64_t g_size = (num_index[i]+n_warp - 1) / n_warp;
    

    if(cpu_buffer_flag == false){
      read_feature_kernel<TYPE><<<g_size, b_size, 0, streams[i] >>>(a->d_array_ptr, tensor_ptr,
                                                    index_ptr, dim, num_index[i], cache_dim, 0);
    }
    else{
      read_feature_kernel_with_cpu_backing_memory<<<g_size, b_size, 0, streams[i] >>>(a->d_array_ptr, d_range ,tensor_ptr,
                                                    index_ptr, dim, num_index[i], cache_dim, CPU_buffer, seq_flag, 
                                                    d_cpu_access, 0);
    }
    total_access += num_index[i];
  }

  for (int i = 0; i < num_iter; i++) {
    cudaStreamSynchronize(streams[i]);
  }

  cuda_err_chk(cudaDeviceSynchronize());
  cuda_err_chk(cudaDeviceSynchronize());
  cudaMemcpy(&cpu_access_count, d_cpu_access, sizeof(unsigned int), cudaMemcpyDeviceToHost);

  auto t2 = Clock::now();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(
      t2 - t1); // Microsecond (as int)
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      t2 - t1); // Microsecond (as int)
  const float ms_fractional =
      static_cast<float>(us.count()) / 1000; // Milliseconds (as float)

  //std::cout << "Duration = " << us.count() << "µs (" << ms_fractional << "ms)"
    //        << std::endl;
 
  kernel_time += ms_fractional;

  for (int i = 0; i < num_iter; i++) {
      cudaStreamDestroy(streams[i]);
  }
  return;
}


template <typename TYPE>
void BAM_Feature_Store<TYPE>::read_feature_merged_hetero(int num_iter, const std::vector<uint64_t>&  i_ptr_list, const std::vector<uint64_t>&  i_index_ptr_list,
                                     const std::vector<uint64_t>&   num_index, int dim, int cache_dim, const std::vector<uint64_t>& key_off) {

  cudaStream_t streams[num_iter];
  for (int i = 0; i < num_iter; i++) {
      cudaStreamCreate(&streams[i]);
  }

  cuda_err_chk(cudaDeviceSynchronize());
  auto t1 = Clock::now();

  for(uint64_t i = 0;  i < num_iter; i++){
    uint64_t i_ptr = i_ptr_list[i];
    uint64_t    i_index_ptr =  i_index_ptr_list[i];         
    TYPE *tensor_ptr = (TYPE *) i_ptr;
    int64_t *index_ptr = (int64_t *)i_index_ptr;

    uint64_t b_size = blkSize;
    uint64_t n_warp = b_size / 32;
    uint64_t g_size = (num_index[i]+n_warp - 1) / n_warp;
    

    if(cpu_buffer_flag == false){
      read_feature_kernel<TYPE><<<g_size, b_size, 0, streams[i] >>>(a->d_array_ptr, tensor_ptr,
                                                    index_ptr, dim, num_index[i], cache_dim, key_off[i]);
    }
    else{
      read_feature_kernel_with_cpu_backing_memory<<<g_size, b_size, 0, streams[i] >>>(a->d_array_ptr, d_range ,tensor_ptr,
                                                    index_ptr, dim, num_index[i], cache_dim, CPU_buffer, seq_flag, 
                                                    d_cpu_access, key_off[i]);
    }
    total_access += num_index[i];
  }

  for (int i = 0; i < num_iter; i++) {
    cudaStreamSynchronize(streams[i]);
  }

  cuda_err_chk(cudaDeviceSynchronize());
  cuda_err_chk(cudaDeviceSynchronize());
  cudaMemcpy(&cpu_access_count, d_cpu_access, sizeof(unsigned int), cudaMemcpyDeviceToHost);

  auto t2 = Clock::now();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(
      t2 - t1); // Microsecond (as int)
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      t2 - t1); // Microsecond (as int)
  const float ms_fractional =
      static_cast<float>(us.count()) / 1000; // Milliseconds (as float)

  //std::cout << "Duration = " << us.count() << "µs (" << ms_fractional << "ms)"
    //        << std::endl;
 
  kernel_time += ms_fractional;

  for (int i = 0; i < num_iter; i++) {
      cudaStreamDestroy(streams[i]);
  }
  return;
}




template <typename TYPE>
void BAM_Feature_Store<TYPE>::SA_read_feature(uint64_t i_ptr, uint64_t i_index_ptr,
                                     int64_t num_index, int dim, int cache_dim, uint64_t key_off, uint64_t i_static_info_ptr) {


  TYPE *tensor_ptr = (TYPE *)i_ptr;
  int64_t *index_ptr = (int64_t *)i_index_ptr;
  uint8_t* static_info_ptr = (uint8_t*) i_static_info_ptr;

  uint64_t b_size = 128;
  uint64_t n_warp = b_size / 32;
  uint64_t g_size = (num_index+n_warp - 1) / n_warp;

  cuda_err_chk(cudaDeviceSynchronize());
  auto t1 = Clock::now();

  SA_read_feature_kernel<TYPE><<<g_size, b_size>>>(cache_ptr, tensor_ptr,
                                                  index_ptr, dim, num_index, cache_dim, key_off, head_ptr, static_info_ptr, update_counter);
  // if(cpu_buffer_flag == false){
  //   read_feature_kernel<TYPE><<<g_size, b_size>>>(a->d_array_ptr, tensor_ptr,
  //                                                 index_ptr, dim, num_index, cache_dim, key_off);
  // }
  // else{
  //   read_feature_kernel_with_cpu_backing_memory<<<g_size, b_size>>>(a->d_array_ptr, d_range, tensor_ptr,
  //                                                 index_ptr, dim, num_index, cache_dim, CPU_buffer, seq_flag,
  //                                                 d_cpu_access, key_off);
  // }
  cuda_err_chk(cudaDeviceSynchronize());
  auto t2 = Clock::now();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(
      t2 - t1); // Microsecond (as int)
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      t2 - t1); // Microsecond (as int)
  const float ms_fractional =
      static_cast<float>(us.count()) / 1000; // Milliseconds (as float)
  printf("DEVICE: %llu kernel time for iteration:%f\n", (unsigned long long)cudaDevice, (float) ms_fractional);
  cudaMemcpy(&cpu_access_count, d_cpu_access, sizeof(unsigned int), cudaMemcpyDeviceToHost);

  kernel_time += ms_fractional;
  total_access += num_index;

  update_counter += 1;

  return;
}



template <typename TYPE>
void BAM_Feature_Store<TYPE>::SA_read_feature_dist( const std::vector<uint64_t>&  i_return_ptr_list, const std::vector<uint64_t>&  i_index_ptr_list,
const std::vector<uint64_t>&  index_size_list, int num_gpu, int dim, int cache_dim, uint64_t key_off, const std::vector<uint64_t>&  i_static_info_ptr_list) {


  bool need_static = false;
  if(eviction_policy == 1 || eviction_policy == 3) need_static = true;

  cudaStream_t streams[num_gpu];
  for (int i = 0; i < num_gpu; i++) {
      cudaStreamCreate(&streams[i]);
  }
  
  cuda_err_chk(cudaDeviceSynchronize());
  auto t1 = Clock::now();

  for (int i = 0; i < num_gpu; i++) {

      TYPE *tensor_ptr = (TYPE *)(i_return_ptr_list[i]);
      int64_t *index_ptr = (int64_t *)(i_index_ptr_list[i]);
      uint64_t index_size = index_size_list[i];

      uint8_t* static_info_ptr = nullptr;
      if(need_static){
        static_info_ptr = (uint8_t*) i_static_info_ptr_list[i];
      }

      //printf("index size: %llu\n", index_size);

      uint64_t b_size = 128;
      uint64_t n_warp = b_size / 32;
      uint64_t g_size = (index_size+n_warp - 1) / n_warp;
    //auto t1 = Clock::now();

      if(use_PVP){
      //  printf("PVP read feature\n");
        SA_read_feature_kernel_with_PVP<TYPE><<<g_size, b_size, 0, streams[i]>>>(cache_ptr, tensor_ptr,
                                                index_ptr, node_flag_buffer, PVP_pinned_data, dim, index_size, cache_dim, i, key_off, head_ptr, static_info_ptr, update_counter, debug_mode, prefetch_counter);
      }                
      else{
        SA_read_feature_kernel<TYPE><<<g_size, b_size, 0, streams[i]>>>(cache_ptr, tensor_ptr,
                                                index_ptr, dim, index_size, cache_dim, key_off, head_ptr, static_info_ptr, update_counter);
      } 

      total_access += index_size;
  }

  cuda_err_chk(cudaDeviceSynchronize());
  for (int i = 0; i < num_gpu; i++) {
      cudaStreamDestroy(streams[i]);
  }

  auto t2 = Clock::now();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(
      t2 - t1); // Microsecond (as int)
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      t2 - t1); // Microsecond (as int)
  const float ms_fractional =
      static_cast<float>(us.count()) / 1000; // Milliseconds (as float)

  kernel_time += ms_fractional;

  //printf("kernel time for iteration:%f\n", (float) ms_fractional);

  if(use_PVP){
    for(int i  = 0; i < n_gpus; i++){
          cudaMemset(node_flag_buffer_array[i], 0, sizeof(uint64_t) * max_sample_size);
        }
  }

  update_counter += 1;

  return;
}



template <typename TYPE>
void BAM_Feature_Store<TYPE>::gather_feature_list(uint64_t i_return_ptr, const std::vector<uint64_t>&  i_return_ptr_list, const std::vector<uint64_t>&  index_size_list, 
                                            int num_gpu, int dim, int my_rank, const std::vector<uint64_t>&  i_meta_buffer){

  TYPE* final_tensor_ptr = (TYPE *)i_return_ptr;
 cudaStream_t streams[num_gpu];
  for (int i = 0; i < num_gpu; i++) {
      cudaStreamCreate(&streams[i]);
  }
  
  cuda_err_chk(cudaDeviceSynchronize());
  auto t1 = Clock::now();

  for (int i = 0; i < num_gpu; i++) {
    TYPE* src_tensor_ptr = (TYPE *)(i_return_ptr_list[i]);
    uint64_t index_size = index_size_list[i];

    uint64_t* d_meta_buffer_ptr =  (uint64_t* ) (i_meta_buffer[i]);

    uint64_t b_size = 256;
    //uint64_t n_warp = b_size / 32;
    //uint64_t g_size = (index_size+n_warp - 1) / n_warp;
    uint64_t g_size = index_size;
    gather_feature_kernel<TYPE><<<g_size, b_size, 0, streams[i]>>>(final_tensor_ptr, src_tensor_ptr,
                                                d_meta_buffer_ptr, dim, index_size, i, my_rank);

  }

  cuda_err_chk(cudaDeviceSynchronize());
  for (int i = 0; i < num_gpu; i++) {
      cudaStreamDestroy(streams[i]);
  }
}


template <typename TYPE>
void BAM_Feature_Store<TYPE>::gather_feature_list_hetero(uint64_t i_return_ptr, const std::vector<uint64_t>&  i_return_ptr_list, const std::vector<uint64_t>&  index_size_list, 
                                            int num_gpu, int dim, int my_rank, const std::vector<uint64_t>&  i_meta_buffer){

  TYPE** final_tensor_ptr = (TYPE **)i_return_ptr;
 cudaStream_t streams[num_gpu];
  for (int i = 0; i < num_gpu; i++) {
      cudaStreamCreate(&streams[i]);
  }
  
  cuda_err_chk(cudaDeviceSynchronize());
  auto t1 = Clock::now();

  for (int i = 0; i < num_gpu; i++) {
    TYPE* src_tensor_ptr = (TYPE *)(i_return_ptr_list[i]);
    uint64_t index_size = index_size_list[i];

    uint64_t* d_meta_buffer_ptr =  (uint64_t* ) (i_meta_buffer[i]);

    uint64_t b_size = 256;
    //uint64_t n_warp = b_size / 32;
    //uint64_t g_size = (index_size+n_warp - 1) / n_warp;
    uint64_t g_size = index_size;
    gather_feature_hetero_kernel<TYPE><<<g_size, b_size, 0, streams[i]>>>(final_tensor_ptr, src_tensor_ptr,
                                                d_meta_buffer_ptr, dim, index_size, i, my_rank);

  }

  cuda_err_chk(cudaDeviceSynchronize());
  for (int i = 0; i < num_gpu; i++) {
      cudaStreamDestroy(streams[i]);
  }
}





template <typename TYPE>
void BAM_Feature_Store<TYPE>::split_node_list_init(uint64_t i_index_ptr, int64_t num_gpu,
                                              int64_t index_size, uint64_t i_index_pointer_list){
    int64_t* index_ptr = (int64_t *)i_index_ptr;
    uint64_t* index_pointer_list = (uint64_t *) i_index_pointer_list;
    
    size_t g_size = (index_size + 1023)/1024;

    split_node_list_init_kernel<TYPE><<<g_size,1024>>>(index_ptr, index_pointer_list, num_gpu, index_size);
    cuda_err_chk(cudaDeviceSynchronize());
}


// template <typename TYPE>
// void BAM_Feature_Store<TYPE>::split_node_list_init2(uint64_t i_index_ptr, int64_t num_gpu,
//                                               int64_t index_size, uint64_t i_index_pointer_list){
//     int64_t* index_ptr = (int64_t *)i_index_ptr;
//     uint64_t* index_pointer_list = (uint64_t *) i_index_pointer_list;
    
//     size_t g_size = (index_size + 1023)/1024;

//     split_node_list_init_kernel2<TYPE><<<g_size,1024>>>(index_ptr, index_pointer_list, num_gpu, index_size);
//     cuda_err_chk(cudaDeviceSynchronize());
// }




template <typename TYPE>
void BAM_Feature_Store<TYPE>::split_node_list_init_hetero(const std::vector<uint64_t>& index_ptr_list, int64_t num_gpu,
                                              const std::vector<uint64_t>& index_size_list, uint64_t i_index_pointer_list){
    
    int num_feat = index_ptr_list.size();

    cudaStream_t streams[num_feat];
    for (int i = 0; i < num_feat; i++) {
      cudaStreamCreate(&streams[i]);
    }

    uint64_t off = 0;
    uint64_t* index_pointer_list = (uint64_t *) i_index_pointer_list;

    for (int i = 0; i < num_feat; i++) {
      int64_t* index_ptr = (int64_t *)(index_ptr_list[i]);
      uint64_t index_size = index_size_list[i];
      size_t g_size = (index_size + 1023)/1024;

      split_node_list_init_kernel<TYPE><<<g_size,1024, 0, streams[i]>>>(index_ptr, index_pointer_list, num_gpu, index_size);

      off += index_size;
    }
  

    cuda_err_chk(cudaDeviceSynchronize());
    for (int i = 0; i < num_feat; i++) {
        cudaStreamDestroy(streams[i]);
    }
    
}

template <typename TYPE>
void BAM_Feature_Store<TYPE>::split_node_list(uint64_t i_index_ptr, int64_t num_gpu, 
                                              int64_t index_size, uint64_t i_bucket_ptr_list, uint64_t i_index_pointer_list,
                                              uint64_t i_meta_buffer){
    int64_t* index_ptr = (int64_t *)i_index_ptr;
    uint64_t* index_pointer_list = (uint64_t *) i_index_pointer_list;
    uint64_t* bucket_ptr_list = (uint64_t *) i_bucket_ptr_list;

    uint64_t* meta_buffer_list_ptr = (uint64_t*) i_meta_buffer;

    size_t g_size = (index_size + 1023)/1024;

    split_node_list_kernel<TYPE><<<g_size,1024>>>(index_ptr, bucket_ptr_list, index_pointer_list, num_gpu, index_size, meta_buffer_list_ptr);
    cuda_err_chk(cudaDeviceSynchronize());
}

// template <typename TYPE>
// void BAM_Feature_Store<TYPE>::split_node_list2(uint64_t i_index_ptr, int64_t num_gpu, 
//                                               int64_t index_size, uint64_t i_bucket_ptr_list, uint64_t i_index_pointer_list,
//                                               uint64_t i_meta_buffer){
//     int64_t* index_ptr = (int64_t *)i_index_ptr;
//     uint64_t* index_pointer_list = (uint64_t *) i_index_pointer_list;
//     uint64_t* bucket_ptr_list = (uint64_t *) i_bucket_ptr_list;

//     uint64_t* meta_buffer_list_ptr = (uint64_t*) i_meta_buffer;

//     size_t g_size = (index_size + 1023)/1024;

//     split_node_list_kernel2<TYPE><<<g_size,1024>>>(index_ptr, bucket_ptr_list, index_pointer_list, num_gpu, index_size, meta_buffer_list_ptr);
//     cuda_err_chk(cudaDeviceSynchronize());
// }

template <typename TYPE>
void BAM_Feature_Store<TYPE>::split_node_list_hetero(const std::vector<uint64_t>& index_ptr_list, int64_t num_gpu, 
                                              const std::vector<uint64_t>& index_size_list, uint64_t i_bucket_ptr_list, uint64_t i_index_pointer_list,
                                              uint64_t i_meta_buffer){

    int num_feat = index_ptr_list.size();

    cudaStream_t streams[num_feat];
    for (int i = 0; i < num_feat; i++) {
      cudaStreamCreate(&streams[i]);
    }
                                        


    uint64_t* index_pointer_list = (uint64_t *) i_index_pointer_list;
    uint64_t* meta_buffer_list_ptr = (uint64_t*) i_meta_buffer;
    uint64_t* bucket_ptr_list = (uint64_t *) i_bucket_ptr_list;

    for (int i = 0; i < num_feat; i++) {
      int64_t* index_ptr = (int64_t *)(index_ptr_list[i]);
      uint64_t index_size = index_size_list[i];
      size_t g_size = (index_size + 1023)/1024;

      split_node_list_hetero_kernel<TYPE><<<g_size,1024, 0, streams[i] >>>(index_ptr, bucket_ptr_list, index_pointer_list, num_gpu, index_size, meta_buffer_list_ptr, i);

     // off += index_size;
    }

    cuda_err_chk(cudaDeviceSynchronize());
    for (int i = 0; i < num_feat; i++) {
        cudaStreamDestroy(streams[i]);
    }

}


template <typename TYPE>
void BAM_Feature_Store<TYPE>::print_meta_buffer(const std::vector<uint64_t>&  index_size_list, int num_gpu, int rank ){

  for(int i = 0; i < num_gpu; i++){
    uint64_t meta_len = index_size_list[i];
    print_meta_buffer_kernel<TYPE><<<1,1>>>(d_meta_buffer, i, meta_len, rank);
  }
    cuda_err_chk(cudaDeviceSynchronize());

}




template <typename TYPE>
void BAM_Feature_Store<TYPE>::reset_node_counter(const std::vector<uint64_t>&  i_index_pointer_list, int num_gpu){

    for(int i = 0; i < num_gpu; i++){
      uint64_t i_add = i_index_pointer_list[i];
      void* add = (void*) i_add;
      cudaMemset(add, 0, sizeof(int64_t));
    }
}


// template <typename TYPE>
// void BAM_Feature_Store<TYPE>::reset_node_counter2(uint64_t  i_index_pointer_list, int num_gpu){
//       void* add = (void*) i_index_pointer_list;
//       cudaMemset(add, 0, sizeof(int64_t) * num_gpu);
// }



template <typename TYPE>
void BAM_Feature_Store<TYPE>::create_meta_buffer(uint64_t num_gpu, uint64_t max_size){
  for(int i = 0; i < num_gpu; i++){
    cudaMalloc(&(meta_buffer[i]), sizeof(uint64_t) * max_size);
  }
  cudaMalloc(&d_meta_buffer, sizeof(uint64_t*) * num_gpu);
  cudaMemcpy(d_meta_buffer, meta_buffer, sizeof(uint64_t*) * num_gpu, cudaMemcpyHostToDevice);
}
                                      


// PVP
// BUFFER FORMAT: [GPU0-WB0, GPU1-WB0, ... GPU8-WB0, GPU0-WB1]
template <typename TYPE>
void BAM_Feature_Store<TYPE>::update_reuse_counters(uint64_t batch_array_idx, uint64_t batch_size_idx, uint32_t max_batch_size,
 int num_gpus, int num_buffers) {

  if (update_counter == refresh_time) {
    update_counter = 0;
    //printf("GPU ID:%llu update reuse counter start\n", SA_handle -> my_GPU_id_);
    cudaStreamCreateWithPriority (&update_stream,cudaStreamNonBlocking, low_priority);

    //cuda_err_chk(cudaDeviceSynchronize());
    SA_handle -> flush_next_reuse(update_stream);
  // SA_handle -> flush_next_reuse( transfer_stream);

    //cuda_err_chk(cudaDeviceSynchronize());
    //printf("GPU ID:%llu Flush done \n", SA_handle -> my_GPU_id_);

    auto t1 = Clock::now();


    uint64_t** batch_array_ptr = (uint64_t**) batch_array_idx;
    uint64_t* batch_size_ptr = (uint64_t*) batch_size_idx;


    uint64_t b_size = 128;
    uint64_t n_warp = b_size / 32;
    uint32_t g_x = (max_batch_size + n_warp - 1)/n_warp;
    uint32_t g_y = num_buffers * num_gpus;
    dim3 g_size (g_x,g_y,1);
    dim3 block_size (b_size, 1, 1);

    //printf("update reuse counters num_gpus:%i g_u:%lu\n", num_gpus, (unsigned long) g_y);

    update_reuse_counters_kernel<TYPE><<<g_size, block_size, 0, update_stream>>>(cache_ptr,
                                                batch_array_ptr, batch_size_ptr, num_gpus);
          
  }                                      

  return;
}


template <typename TYPE>
void BAM_Feature_Store<TYPE>::prefetch_from_victim_queue(){
  cudaStreamCreateWithPriority (&transfer_stream,cudaStreamNonBlocking, low_priority);

  num_evicted_cl =  SA_handle -> prefetch_from_victim_queue(PVP_pinned_data, PVP_pinned_idx, head_ptr, transfer_stream);
  unsigned int GPU = SA_handle ->  my_GPU_id_;

 // printf("GPU: %llu num evicted cl:%llu\n", (unsigned long long) GPU, num_evicted_cl);
  head_ptr = (head_ptr + 1) % wb_size;
}

template <typename TYPE>
void BAM_Feature_Store<TYPE>::fill_batch(){

  //cudaStreamCreateWithFlags(&fill_stream, cudaStreamNonBlocking);
	if(first){
    first = false;
		return;
	}
  // uint64_t** d_node_flag_ptr = (uint64_t**) i_node_flag_ptr;
  uint32_t b_size = 128;
  uint32_t g_size = (num_evicted_cl + b_size - 1)/b_size; 
  cuda_err_chk(cudaDeviceSynchronize());
  

  //fill_batch_kernel<TYPE><<<g_size, b_size, 0, fill_stream>>>(PVP_pinned_idx, d_node_flag_ptr, num_evicted_cl, dim);
  //fill_batch_kernel<TYPE><<<g_size, b_size, 0, fill_stream >>>(PVP_pinned_idx, node_flag_buffer, num_evicted_cl, dim);

  fill_batch_kernel<TYPE><<<g_size, b_size>>>(PVP_pinned_idx, node_flag_buffer, num_evicted_cl, dim, debug_mode, evict_counter, SA_handle->my_GPU_id_, max_sample_size);

  cuda_err_chk(cudaDeviceSynchronize());


  cudaStreamDestroy(transfer_stream);

  if (update_counter == refresh_time) {
    cudaStreamDestroy(update_stream);
  }

  return;
}


//self.BAM_FS.get_static_info(static_info_ten.data_ptr(), index_ten.data_ptr(), len(index_ten))

template <typename TYPE>
void BAM_Feature_Store<TYPE>::get_static_info(uint64_t i_out, uint64_t i_index_ptr, uint64_t index_len){
  uint8_t* out_ptr = (uint8_t*) i_out;
  uint64_t* index_ptr = (uint64_t*) i_index_ptr;

  uint32_t b_size = 128;
  uint32_t g_size = (index_len + b_size - 1)/b_size; 

  get_static_info_kernel<<<g_size, b_size>>>(out_ptr, index_ptr, index_len, d_static_val_array);
  cuda_err_chk(cudaDeviceSynchronize());


}

template <typename TYPE>
void BAM_Feature_Store<TYPE>::get_static_info_dist(const std::vector<uint64_t>&  i_out, const std::vector<uint64_t>&  i_index_ptr, const std::vector<uint64_t>&  index_len_array){
  
  cudaStream_t streams[n_gpus];
  for (int i = 0; i < n_gpus; i++) {
      cudaStreamCreate(&streams[i]);
  }
  for(int i = 0; i < n_gpus;i++){ 
    
    uint8_t* out_ptr = (uint8_t*) i_out[i];
    uint64_t* index_ptr = (uint64_t*) i_index_ptr[i];
    uint64_t index_len = index_len_array[i];

    uint32_t b_size = 128;
    uint32_t g_size = (index_len + b_size - 1)/b_size; 

    get_static_info_kernel<<<g_size, b_size, 0, streams[i]>>>(out_ptr, index_ptr, index_len, d_static_val_array);
  }

   for (int i = 0; i < n_gpus; i++) {
      cudaStreamSynchronize(streams[i]);
  }

   
}


template <typename TYPE>
void  BAM_Feature_Store<TYPE>::store_tensor(uint64_t tensor_ptr, uint64_t num, uint64_t offset){

  TYPE* t_ptr = (TYPE*) tensor_ptr;
  
  page_cache_d_t* d_pc = (page_cache_d_t*) (h_pc -> d_pc_ptr);

  size_t g_size = (num + 1023)/1024;
  printf("g size: %i num: %llu\n", g_size, num);
  write_feature_kernel<TYPE><<<g_size, 1024>>>(h_pc->pdt.d_ctrls, d_pc, a->d_array_ptr, t_ptr, 4096, offset);
  cuda_err_chk(cudaDeviceSynchronize());
  printf("CALLLING FLUSH\n");
  h_pc->flush_cache();
  cuda_err_chk(cudaDeviceSynchronize());

}


template <typename TYPE>
void  BAM_Feature_Store<TYPE>::flush_cache(){
  h_pc->flush_cache();
  cuda_err_chk(cudaDeviceSynchronize());
}



template <typename TYPE>
void  BAM_Feature_Store<TYPE>::set_cpu_buffer(uint64_t idx_buffer, int num){

  int bsize = 1024;
  int grid = (num + bsize - 1) / bsize;
  uint64_t* idx_ptr = (uint64_t* ) idx_buffer;
  set_cpu_buffer_kernel<TYPE><<<grid,bsize>>>(d_range, idx_ptr, num, pageSize);
  cuda_err_chk(cudaDeviceSynchronize());
  seq_flag = false;


}



template <typename TYPE>
void  BAM_Feature_Store<TYPE>::set_offsets(uint64_t in_off, uint64_t index_off, uint64_t data_off){

 offset_array = new uint64_t[3];
    printf("set offset: in_off: %llu index_off: %llu data_off: %llu offset_ptr:%llu\n", in_off, index_off, data_off, (uint64_t) offset_array);

  offset_array[0] = (in_off);
  offset_array[1] = (index_off);
  offset_array[2] = (data_off);

}


template <typename TYPE>
uint64_t BAM_Feature_Store<TYPE>::get_offset_array(){
  return ((uint64_t) offset_array);
}

template <typename TYPE>
uint64_t BAM_Feature_Store<TYPE>::get_array_ptr(){
	return ((uint64_t) (a->d_array_ptr));
}


template <typename TYPE>
void  BAM_Feature_Store<TYPE>::read_tensor(uint64_t num, uint64_t offset){
  read_kernel<TYPE><<<1, 1>>>(a->d_array_ptr, num, offset);
  cuda_err_chk(cudaDeviceSynchronize());

}


template <typename TYPE>
unsigned int BAM_Feature_Store<TYPE>::get_cpu_access_count(){
	return cpu_access_count;
}

template <typename TYPE>
void BAM_Feature_Store<TYPE>::flush_cpu_access_count(){
	cpu_access_count = 0;
  cudaMemset(d_cpu_access, 0 , sizeof(unsigned));
}

template <typename T>
BAM_Feature_Store<T> create_BAM_Feature_Store() {
    return BAM_Feature_Store<T>();
}




template <typename TYPE>
void BAM_Feature_Store<TYPE>::print_victim_buffer_index(uint64_t offset, uint64_t len) {
  SA_handle->print_victim_buffer_index(offset, len); 
  return;
}


template <typename TYPE>
void BAM_Feature_Store<TYPE>::print_victim_buffer_data(uint64_t offset, uint64_t len) {
  SA_handle->print_victim_buffer_data(offset, len); 
  return;
}


void Emulate_SA::read_feature(uint64_t i_ptr, uint64_t i_index_ptr,
		                                     int64_t num_index, int dim, int cache_dim, uint64_t key_off, uint64_t i_static_info_ptr) {
                              
	  float *tensor_ptr = (float *)i_ptr;
	    int64_t *index_ptr = (int64_t *)i_index_ptr;
	      uint8_t* static_info_ptr = (uint8_t*) i_static_info_ptr;

	        uint64_t b_size = 128;
		  uint64_t n_warp = b_size / 32;
		    uint64_t g_size = (num_index+n_warp - 1) / n_warp;

		      cuda_err_chk(cudaDeviceSynchronize());
		        auto t1 = Clock::now();

			  Emulate_SA_read_feature_kernel<float><<<g_size, b_size>>>(Emul_cache_ptr, tensor_ptr,
					                                                  index_ptr, dim, num_index, cache_dim, key_off, static_info_ptr);

      cuda_err_chk(cudaDeviceSynchronize());
}

void Emulate_SA::read_feature_with_color(uint64_t i_ptr, uint64_t i_index_ptr,
		                                     int64_t num_index, int dim, int cache_dim, uint64_t key_off, uint64_t i_static_info_ptr, uint64_t i_color_ptr) {
                              
	  float *tensor_ptr = (float *)i_ptr;
	    int64_t *index_ptr = (int64_t *)i_index_ptr;
	      uint8_t* static_info_ptr = (uint8_t*) i_static_info_ptr;

	        uint64_t b_size = 128;
		  uint64_t n_warp = b_size / 32;
		    uint64_t g_size = (num_index+n_warp - 1) / n_warp;

		      cuda_err_chk(cudaDeviceSynchronize());
		        auto t1 = Clock::now();

          uint64_t* color_data = (uint64_t*) i_color_ptr;

			  Emulate_SA_read_feature_with_color_kernel<float><<<g_size, b_size>>>(Emul_cache_ptr, tensor_ptr,
					                                                  index_ptr, dim, num_index, cache_dim, key_off, static_info_ptr, color_data);

      cuda_err_chk(cudaDeviceSynchronize());
}


void Emulate_SA::init_cache(uint64_t num_sets, uint64_t num_ways, uint64_t page_size, uint32_t cudaDevice, uint8_t eviction_policy, int cache_track, uint64_t num_colors){
    bool c_track = false;
    if(cache_track == 1)
      c_track = true;
	  Emul_SA_handle = new Emulate_SA_handle<float>(num_sets, num_ways, page_size, cudaDevice, eviction_policy, c_track, num_colors);
	  Emul_cache_ptr = Emul_SA_handle -> get_ptr();
}

void Emulate_SA::print_counters(){
  printf("Printing Cache Counters\n");
  //Emul_SA_handle -> print_counters();
  Emulate_SA_print_counters<<<1,1>>>(Emul_cache_ptr);
}

float Emulate_SA::color_score(uint64_t color){
  return Emul_SA_handle -> color_score(color);
}


PYBIND11_MODULE(BAM_Feature_Store, m) {
  m.doc() = "Python bindings for an example library";

  namespace py = pybind11;

  //py::class_<BAM_Feature_Store<>, std::unique_ptr<BAM_Feature_Store<float>, py::nodelete>>(m, "BAM_Feature_Store")
    py::class_<BAM_Feature_Store<float>>(m, "BAM_Feature_Store_float")
      .def(py::init<>())
      .def("init_controllers", &BAM_Feature_Store<float>::init_controllers)
      .def("init_set_associative_cache", &BAM_Feature_Store<float>::init_set_associative_cache)
      .def("read_feature", &BAM_Feature_Store<float>::read_feature)
      .def("read_feature_hetero", &BAM_Feature_Store<float>::read_feature_hetero)
      .def("read_feature_merged_hetero", &BAM_Feature_Store<float>::read_feature_merged_hetero)
      .def("read_feature_merged", &BAM_Feature_Store<float>::read_feature_merged)

      .def("SA_read_feature", &BAM_Feature_Store<float>::SA_read_feature)
      .def("SA_read_feature_dist", &BAM_Feature_Store<float>::SA_read_feature_dist)

      .def("split_node_list_init", &BAM_Feature_Store<float>::split_node_list_init)
      .def("split_node_list_init_hetero", &BAM_Feature_Store<float>::split_node_list_init_hetero)

      .def("split_node_list", &BAM_Feature_Store<float>::split_node_list)
      .def("split_node_list_hetero", &BAM_Feature_Store<float>::split_node_list_hetero)

      .def("reset_node_counter", &BAM_Feature_Store<float>::reset_node_counter)
      .def("create_meta_buffer", &BAM_Feature_Store<float>::create_meta_buffer)
      .def("gather_feature_list", &BAM_Feature_Store<float>::gather_feature_list)
      .def("gather_feature_list_hetero", &BAM_Feature_Store<float>::gather_feature_list_hetero)

      .def("update_reuse_counters", &BAM_Feature_Store<float>::update_reuse_counters)

      .def("set_window_buffering", &BAM_Feature_Store<float>::set_window_buffering)
      .def("cpu_backing_buffer", &BAM_Feature_Store<float>::cpu_backing_buffer)
      .def("set_cpu_buffer", &BAM_Feature_Store<float>::set_cpu_buffer)

      .def("flush_cache", &BAM_Feature_Store<float>::flush_cache)
      .def("store_tensor",  &BAM_Feature_Store<float>::store_tensor)
      .def("read_tensor",  &BAM_Feature_Store<float>::read_tensor)

      .def("get_array_ptr", &BAM_Feature_Store<float>::get_array_ptr)
      .def("get_offset_array", &BAM_Feature_Store<float>::get_offset_array)
      .def("set_offsets", &BAM_Feature_Store<float>::set_offsets)
      .def("get_cpu_access_count", &BAM_Feature_Store<float>::get_cpu_access_count)
      .def("flush_cpu_access_count", &BAM_Feature_Store<float>::flush_cpu_access_count)

      .def("fill_batch", &BAM_Feature_Store<float>::fill_batch)

      .def("prefetch_from_victim_queue", &BAM_Feature_Store<float>::prefetch_from_victim_queue)


      .def("print_victim_buffer_index", &BAM_Feature_Store<float>::print_victim_buffer_index)
      .def("print_victim_buffer_data", &BAM_Feature_Store<float>::print_victim_buffer_data)
      .def("get_static_info", &BAM_Feature_Store<float>::get_static_info)
      .def("get_static_info_dist", &BAM_Feature_Store<float>::get_static_info_dist)

      .def("create_static_info_buffer", &BAM_Feature_Store<float>::create_static_info_buffer)

      .def("print_stats", &BAM_Feature_Store<float>::print_stats)
      .def("print_stats_rank", &BAM_Feature_Store<float>::print_stats_rank)

      .def("print_meta_buffer", &BAM_Feature_Store<float>::print_meta_buffer);




    py::class_<BAM_Feature_Store<int64_t>>(m, "BAM_Feature_Store_long")
      .def(py::init<>())
      .def("init_controllers", &BAM_Feature_Store<int64_t>::init_controllers)
      .def("init_set_associative_cache", &BAM_Feature_Store<int64_t>::init_set_associative_cache)
      .def("read_feature", &BAM_Feature_Store<int64_t>::read_feature)
      .def("read_feature_hetero", &BAM_Feature_Store<int64_t>::read_feature_hetero)

      .def("read_feature_merged", &BAM_Feature_Store<int64_t>::read_feature_merged)
      .def("read_feature_merged_hetero", &BAM_Feature_Store<int64_t>::read_feature_merged_hetero)

      .def("SA_read_feature", &BAM_Feature_Store<int64_t>::SA_read_feature)
      .def("SA_read_feature_dist", &BAM_Feature_Store<int64_t>::SA_read_feature_dist)


      .def("split_node_list_init", &BAM_Feature_Store<int64_t>::split_node_list_init)
      .def("split_node_list_init_hetero", &BAM_Feature_Store<int64_t>::split_node_list_init_hetero)

      .def("split_node_list", &BAM_Feature_Store<int64_t>::split_node_list)
      .def("split_node_list_hetero", &BAM_Feature_Store<int64_t>::split_node_list_hetero)

      
      .def("reset_node_counter", &BAM_Feature_Store<int64_t>::reset_node_counter)
      .def("create_meta_buffer", &BAM_Feature_Store<int64_t>::create_meta_buffer)
      .def("gather_feature_list", &BAM_Feature_Store<int64_t>::gather_feature_list)
      .def("gather_feature_list_hetero", &BAM_Feature_Store<int64_t>::gather_feature_list_hetero)

      
      .def("update_reuse_counters", &BAM_Feature_Store<int64_t>::update_reuse_counters)


      .def("set_window_buffering", &BAM_Feature_Store<int64_t>::set_window_buffering)
      .def("cpu_backing_buffer", &BAM_Feature_Store<int64_t>::cpu_backing_buffer)
      .def("set_cpu_buffer", &BAM_Feature_Store<int64_t>::set_cpu_buffer)

      .def("flush_cache", &BAM_Feature_Store<int64_t>::flush_cache)
      .def("store_tensor",  &BAM_Feature_Store<int64_t>::store_tensor)
      .def("read_tensor",  &BAM_Feature_Store<int64_t>::read_tensor)

      .def("get_array_ptr", &BAM_Feature_Store<int64_t>::get_array_ptr)
      .def("get_offset_array", &BAM_Feature_Store<int64_t>::get_offset_array)
      .def("set_offsets", &BAM_Feature_Store<int64_t>::set_offsets)
      .def("get_cpu_access_count", &BAM_Feature_Store<int64_t>::get_cpu_access_count)
      .def("flush_cpu_access_count", &BAM_Feature_Store<int64_t>::flush_cpu_access_count)

      .def("prefetch_from_victim_queue", &BAM_Feature_Store<int64_t>::prefetch_from_victim_queue)
      .def("fill_batch", &BAM_Feature_Store<int64_t>::fill_batch)

      .def("print_victim_buffer_index", &BAM_Feature_Store<int64_t>::print_victim_buffer_index)
      .def("print_victim_buffer_data", &BAM_Feature_Store<int64_t>::print_victim_buffer_data)

      .def("get_static_info", &BAM_Feature_Store<int64_t>::get_static_info)
      .def("get_static_info_dist", &BAM_Feature_Store<int64_t>::get_static_info_dist)
      .def("create_static_info_buffer", &BAM_Feature_Store<int64_t>::create_static_info_buffer)

      .def("print_stats", &BAM_Feature_Store<int64_t>::print_stats)
          .def("print_stats_rank", &BAM_Feature_Store<int64_t>::print_stats_rank)
      .def("print_meta_buffer", &BAM_Feature_Store<int64_t>::print_meta_buffer);

      




      py::class_<GIDS_Controllers>(m, "GIDS_Controllers")
      .def(py::init<>())
      .def("init_GIDS_controllers", &GIDS_Controllers::init_GIDS_controllers);


      py::class_<Emulate_SA>(m, "Emulate_SA")
      .def(py::init<>())
      .def("init_cache", &Emulate_SA::init_cache)
      .def("print_counters", &Emulate_SA::print_counters)
      .def("color_score", &Emulate_SA::color_score)
 			.def("read_feature", &Emulate_SA::read_feature)
      .def("read_feature_with_color", &Emulate_SA::read_feature_with_color);

}



//gids




