#ifndef __NVSHMEM_CACHE_CU__
#define __NVSHMEM_CACHE_CU__

#include <cuda.h>


#include "nvshmem_cache.h"
#include "nvshmem_cache_kernel.cu"
#include "gids_nvme.cu"
#include <mpi.h>

// #include "nvshmem.h"
// #include "nvshmemx.h"



typedef std::chrono::high_resolution_clock Clock;

template <typename TYPE>
struct NVSHMEM_Cache {

    NVSHMEM_cache_handle<TYPE>* cache_handle;
    NVSHMEM_cache_d_t<TYPE>* cache_ptr;

    uint64_t num_eles;
    uint64_t read_offset = 0;

    uint32_t n_gpus = 1;
    uint32_t n_ctrls = 1;
    uint32_t n_pages = 0;
    uint32_t page_size = 4096;
    uint64_t num_sets = 1;
    uint32_t num_ways = 32;
    uint32_t dim = 32;

    uint32_t cudaDevice = 0;

    size_t blkSize = 128;
    float kernel_time = 0; 
    uint64_t total_access = 0;

   
    std::vector<Controller *> ctrls;

    void init_cache(GIDS_Controllers GIDS_ctrl, uint32_t ps, uint64_t read_off, uint64_t cache_size, uint32_t num_gpus, uint64_t num_ele, uint64_t num_ssd, uint64_t ways);
    void read_feature(uint64_t i_ptr, uint64_t i_index_ptr, int64_t num_index, int dim, int cache_dim);
    void print_stats();
    //MPI functions
    void init_mpi_nvshmem();
    void finalize_mpi();
    int get_world_size();
    int get_rank();

    uint64_t nvshmem_malloc(uint64_t size);

    
};




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

  num_sets = n_pages / num_ways;
  cache_handle = new NVSHMEM_cache_handle<TYPE>(num_sets, num_ways, page_size, ctrls[0][0], ctrls, cudaDevice, cudaDevice, num_gpus);
  cache_ptr = cache_handle -> get_ptr();

  //cuda_err_chk(cudaDeviceSynchronize());
  printf("Init done\n");

  return;
}





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
uint64_t NVSHMEM_Cache<TYPE>::nvshmem_malloc(uint64_t size){
  // int* dest = (int *) nvshmem_malloc(size);
  // uint64_t dest_ptr = (uint64_t) dest;
  // printf("ptr: %p i_ptr:%llu\n", dest, dest_ptr);
  // return dest_ptr;
  return 0;
}


template <typename TYPE>
void NVSHMEM_Cache<TYPE>::init_mpi_nvshmem(){
  int mpi_initialized;
  MPI_Initialized(&mpi_initialized);

  if (!mpi_initialized) {
      MPI_Init(NULL, NULL);  // Initialize MPI with default arguments
          std::cout << "MPI is not initialized correctly!" << std::endl;
  }

  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  nvshmemx_init_attr_t attr;
  attr.mpi_comm = &mpi_comm;
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
  return;
}

template <typename TYPE>
void NVSHMEM_Cache<TYPE>::finalize_mpi() {
    int mpi_finalized;
    MPI_Finalized(&mpi_finalized);
    if (!mpi_finalized) {
        MPI_Finalize();  // Finalize MPI
    }
}

template <typename TYPE>
int NVSHMEM_Cache<TYPE>::get_rank() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

template <typename TYPE>
int NVSHMEM_Cache<TYPE>::get_world_size() {
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
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

 
  std::cout << "Kernel Time: \t " << this->kernel_time << std::endl;
  this->kernel_time = 0;
  std::cout << "Total Access: \t " << this->total_access << std::endl;
  this->total_access = 0;

}


#endif

