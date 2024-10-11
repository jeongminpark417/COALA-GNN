#ifndef __GIDS_NVME_CU__
#define __GIDS_NVME_CU__


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

//#include "gids_kernel.cu"
#include "nvshmem_cache_kernel.cu"




typedef std::chrono::high_resolution_clock Clock;

void GIDS_Controllers::init_GIDS_controllers(uint32_t num_ctrls, uint64_t q_depth, uint64_t num_q, 
                          const std::vector<int>& ssd_list, uint32_t device){

  n_ctrls = num_ctrls;
  queueDepth = q_depth;
  numQueues = num_q;
  cudaDevice = device;
  cudaSetDevice(cudaDevice);

  for (size_t i = 0; i < n_ctrls; i++) {
    ctrls.push_back(new Controller(ctrls_paths[ssd_list[i]], nvmNamespace, cudaDevice, queueDepth, numQueues));
  }
}



template <typename TYPE>
void NVSHMEM_Cache<TYPE>::init_cache(GIDS_Controllers GIDS_ctrl, uint32_t ps, uint64_t read_off, uint64_t cache_size, uint32_t num_gpus, uint64_t num_ele, uint64_t num_ssd, uint64_t ways) {

  num_eles = num_ele;
  read_offset = read_off;
  n_ctrls = num_ssd;
  n_gpus = num_gpus;
  num_ways = ways;

  page_size = ps;
  dim = ps / sizeof(TYPE);

  ctrls = GIDS_ctrl.ctrls;
  cudaDevice = GIDS_ctrl.cudaDevice;


  cudaSetDevice(cudaDevice);

  n_pages = cache_size * 1024LL*1024/page_size;


  std::cout << "n pages: " << (int)(this->n_pages) <<std::endl;
  std::cout << "page size: " << (int)(this->page_size) << std::endl;
  std::cout << "num elements: " << this->num_eles << std::endl;
  std::cout << "cudaDevice" << cudaDevice  << std::endl;
  std::cout << "number of GPUs" << n_gpus  << std::endl;


  num_sets = n_pages / num_ways;
  cache_handle = new NVSHMEM_cache_handle<TYPE>(num_sets, num_ways, page_size, ctrls[0][0], ctrls, cudaDevice, cudaDevice, num_gpus);
  cache_ptr = cache_handle -> get_ptr();

  cuda_err_chk(cudaDeviceSynchronize());
  cuda_err_chk(cudaMalloc(&d_request_counters, sizeof(unsigned int) * n_gpus));

  printf("Init done\n");

  return;
}



///NVSHMEM

template <typename TYPE>
void NVSHMEM_Cache<TYPE>::read_feature(uint64_t i_ptr, uint64_t i_index_ptr,
                                     int64_t num_index, int dim, int cache_dim) {


  TYPE *tensor_ptr = (TYPE *)i_ptr;
  int64_t *index_ptr = (int64_t *)i_index_ptr;

  uint64_t b_size = 128;
  uint64_t n_warp = b_size / 32;
  uint64_t g_size = (num_index+n_warp - 1) / n_warp;

  cuda_err_chk(cudaDeviceSynchronize());
  auto t1 = Clock::now();

  NVShmem_read_feature_kernel<TYPE><<<g_size, b_size>>>(cache_ptr, tensor_ptr,
                                                  index_ptr, dim, num_index, cache_dim);

  cuda_err_chk(cudaDeviceSynchronize());
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
void NVSHMEM_Cache<TYPE>::dist_read_feature(uint64_t i_return_tensor_ptr, uint64_t i_nvshmem_index_ptr, int64_t max_index, int dim, int cache_dim) {
  
  auto t0 = Clock::now();

  cudaStream_t streams[n_gpus];
  for (int i = 0; i < n_gpus; i++) {
      cudaStreamCreate(&streams[i]);
  }

  TYPE *tensor_ptr = (TYPE *) i_return_tensor_ptr;
  int64_t *nvshmem_index_ptr = (int64_t *)i_nvshmem_index_ptr;


  int b_size = 64;
  int ydim = (n_gpus >= 16) ?  16 : n_gpus;
  dim3 b_dim (b_size, ydim, 1);
  uint64_t g_size = (max_index+b_size - 1) / b_size;
  dim3 g_dim (g_size, 1, 1);

  cuda_err_chk(cudaMemset((void*)d_request_counters, 0, sizeof(unsigned int) * n_gpus));
  NVShmem_count_requests_kernel<TYPE><<<g_dim, b_dim>>> (nvshmem_index_ptr, d_request_counters, max_index, n_gpus, rank);

  cuda_err_chk(cudaDeviceSynchronize());

  unsigned int h_request_counters[n_gpus];
  cuda_err_chk(cudaMemcpy(h_request_counters, d_request_counters, sizeof(unsigned int) * n_gpus,  cudaMemcpyDeviceToHost));
  //printf("RANK: %i RECV counters %u %u\n", rank, h_request_counters[0], h_request_counters[1]);
    
  auto t1 = Clock::now();

  for (int i = 0; i < n_gpus; i++) {
    uint64_t b_size = 128;
    uint64_t n_warp = b_size / 32;
    uint64_t g_size = (h_request_counters[i]+n_warp - 1) / n_warp;
   total_access += h_request_counters[i];

    NVShmem_dist_read_feature_kernel<TYPE><<<g_size, b_size, 0, streams[i]>>>(i, cache_ptr, tensor_ptr,
                                                  nvshmem_index_ptr + max_index * 2 * i, dim, h_request_counters[i], cache_dim, rank);
  }

  for (int i = 0; i < n_gpus; i++) {
    cudaStreamSynchronize(streams[i]);
  }
  cuda_err_chk(cudaDeviceSynchronize());

  auto t2 = Clock::now();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(
      t2 - t1); // Microsecond (as int)
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      t2 - t1); // Microsecond (as int)
  const float ms_fractional =
      static_cast<float>(us.count()) / 1000; // Milliseconds (as float)

   kernel_time += ms_fractional;
  cuda_err_chk(cudaDeviceSynchronize());  
  nvshmem_quiet();
  nvshmem_barrier_all();

  us = std::chrono::duration_cast<std::chrono::microseconds>(
      t1 - t0); // Microsecond (as int)
  const float request_ms_fractional =
      static_cast<float>(us.count()) / 1000; // Milliseconds (as float)
 
  request_time += request_ms_fractional;
  request_kernel_time += request_ms_fractional;

  return;
}                            



template <typename TYPE>
void NVSHMEM_Cache<TYPE>::update_NVshmem_metadata(int my_rank, int num_ranks, int pe_node){
  rank = my_rank;
  nranks = num_ranks;
  mype_node = pe_node;
  return;
}


template <typename TYPE>
void NVSHMEM_Cache<TYPE>::send_requests(uint64_t i_src_index_ptr, int64_t num_index, uint64_t i_nvshmem_request_ptr, int max_index){
  auto t0 = Clock::now();

  int64_t *src_index_ptr = (int64_t *)i_src_index_ptr;  
  int64_t* nvshmem_request_ptr = (int64_t *)i_nvshmem_request_ptr;

  int rank1 = nvshmem_my_pe();
  //printf("send request rank: %i\n", rank1);


  uint64_t b_size = 128;
  uint64_t g_size = (num_index+b_size - 1) / b_size;
  //Need to fix for just NVLink domain
  cuda_err_chk(cudaMemset((void*)d_request_counters, 0, sizeof(unsigned int) * n_gpus));
  cuda_err_chk(cudaMemset((void*)nvshmem_request_ptr, -1, sizeof(int64_t) * 2 * max_index * n_gpus));
  cuda_err_chk(cudaDeviceSynchronize());
  nvshmem_barrier_all();
  
  auto t1 = Clock::now();
  send_requests_kernel<<<g_size, b_size>>>(src_index_ptr, num_index, nvshmem_request_ptr + rank * max_index * 2, rank, nranks, d_request_counters);

  cuda_err_chk(cudaDeviceSynchronize());
  nvshmem_quiet();
  nvshmem_barrier_all();

  auto t2 = Clock::now();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(
      t2 - t1); // Microsecond (as int)
  const float ms_fractional =
      static_cast<float>(us.count()) / 1000; // Milliseconds (as float)

  request_kernel_time += ms_fractional;


  us = std::chrono::duration_cast<std::chrono::microseconds>(
      t2 - t0); // Microsecond (as int)
    const float ms_fractional2 =
      static_cast<float>(us.count()) / 1000; // Milliseconds (as float)
  request_time += ms_fractional2;

  // unsigned int h_request_counters[n_gpus];
  // cuda_err_chk(cudaMemcpy(h_request_counters, d_request_counters, sizeof(unsigned int) * n_gpus,  cudaMemcpyDeviceToHost));

  
 // printf("My rank:%i SEND counter vals: %u %u\n", rank, h_request_counters[0], h_request_counters[1]);
  

  return;
}



  template <typename TYPE>
  void NVSHMEM_Cache<TYPE>::print_stats(){

  
    NVShmem_print_kernel<TYPE><<<1,1>>>(cache_ptr);
    cuda_err_chk(cudaDeviceSynchronize())

    for(int i = 0; i < n_ctrls; i++){
    std::cout << "print ctrl reset " << i << ": ";
      (this->ctrls[i])->print_reset_stats();
      std::cout << std::endl;
    }

    std::cout << "Request Time: \t " << this->request_time << std::endl;
    this->request_time = 0;
    std::cout << "Request Kernel Time: \t " << this->request_kernel_time << std::endl;
    this->request_kernel_time = 0;
    std::cout << "Kernel Time: \t " << this->kernel_time << std::endl;
    this->kernel_time = 0;

  
    std::cout << "Total Access: \t " << this->total_access << std::endl;
    this->total_access = 0;

  }



  template <typename TYPE>
  void NVSHMEM_Cache<TYPE>:: init() {
    nvshmem_init();
    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    cudaSetDevice(mype_node);
    printf("PE %d: NVSHMEM initialized and CUDA device set to %d\n", nvshmem_my_pe(), mype_node);
    rank = nvshmem_my_pe();
    nranks = nvshmem_n_pes();

  }

  // Allocate NVSHMEM symmetric memory
  template <typename TYPE>
  uint64_t NVSHMEM_Cache<TYPE>::allocate(int size) {
    void* destination = (void *) nvshmem_malloc(size);
    if (destination == nullptr) {
        fprintf(stderr, "PE %d: nvshmem_malloc failed!\n", nvshmem_my_pe());
        return 0;
    }
    
    uint64_t int_ptr = (uint64_t) destination;
    printf("PE %d: Allocated NVSHMEM memory of size %d at %p and int ptr: %llu\n", nvshmem_my_pe(), size, destination, int_ptr);

    return int_ptr;
  }

  // Free NVSHMEM memory
  template <typename TYPE>
  void NVSHMEM_Cache<TYPE>::free(uint64_t dest_ptr) {
    int* destination = (int*) dest_ptr;
    if (destination) {
        nvshmem_free(destination);
        destination = nullptr;
        printf("PE %d: Freed NVSHMEM memory.\n", nvshmem_my_pe());
    }
  }

  // Finalize NVSHMEM
  template <typename TYPE>
  void NVSHMEM_Cache<TYPE>:: finalize() {
      nvshmem_finalize();
      printf("PE %d: NVSHMEM finalized.\n", nvshmem_my_pe());
  }

  template <typename TYPE>
  int NVSHMEM_Cache<TYPE>:: get_rank() {
      return rank;
  }
  template <typename TYPE>
  int NVSHMEM_Cache<TYPE>:: get_world_size() {
      return nranks;
  }

  template <typename TYPE>
  int NVSHMEM_Cache<TYPE>:: get_mype_node(){
      return mype_node;
  }

  template <typename TYPE>
  void NVSHMEM_Cache<TYPE>:: NVshmem_quiet(){
      nvshmem_quiet();
  }


PYBIND11_MODULE(Dist_Cache, m) {
  m.doc() = "Python bindings for an example library";

  namespace py = pybind11;

      // py::class_<GIDS_Controllers>(m, "Dist_GIDS_Controllers")
      // .def(py::init<>())
      // .def("init_GIDS_controllers", &GIDS_Controllers::init_GIDS_controllers);


      py::class_<NVSHMEM_Cache<float>>(m, "NVSHMEM_Cache")
      .def(py::init<>())
      .def("init_cache", &NVSHMEM_Cache<float>::init_cache)
      .def("read_feature", &NVSHMEM_Cache<float>::read_feature)
      .def("dist_read_feature", &NVSHMEM_Cache<float>::dist_read_feature)
      .def("print_stats", &NVSHMEM_Cache<float>::print_stats)
      .def("update_NVshmem_metadata", &NVSHMEM_Cache<float>::update_NVshmem_metadata)
      .def("send_requests", &NVSHMEM_Cache<float>::send_requests)

      .def("init", &NVSHMEM_Cache<float>::init)
      .def("allocate", &NVSHMEM_Cache<float>::allocate)
      .def("finalize", &NVSHMEM_Cache<float>::finalize)
      .def("free", &NVSHMEM_Cache<float>::free)
      .def("get_rank", &NVSHMEM_Cache<float>::get_rank)
      .def("get_world_size", &NVSHMEM_Cache<float>::get_world_size)
      .def("get_mype_node", &NVSHMEM_Cache<float>::get_mype_node)
      .def("NVshmem_quiet", &NVSHMEM_Cache<float>::NVshmem_quiet);


;

}



//gids



#endif

