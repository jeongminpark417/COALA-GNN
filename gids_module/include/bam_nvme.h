#ifndef BAMNVME_H
#define BAMNVME_H

#include "nvshmem.h"
#include "nvshmemx.h"

#include <cuda.h>
#include "nvshmem_cache.h"
#include <mpi.h>


#include <buffer.h>
#include <cuda.h>
#include <fcntl.h>
#include <nvm_admin.h>
#include <nvm_cmd.h>
#include <nvm_ctrl.h>
#include <nvm_error.h>
#include <nvm_io.h>
#include <nvm_parallel_queue.h>
#include <nvm_queue.h>
#include <nvm_types.h>
#include <nvm_util.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>
#include <util.h>

#include <ctrl.h>
#include <event.h>
#include <page_cache.h>
#include <queue.h>

#include "emulate_set_associative_page_cache.h"


//#include "emulate_set_associative_page_cache.h"


//#define TYPE float
struct Dist_GIDS_Controllers {
  const char *const ctrls_paths[6] = {"/dev/libnvm0","/dev/libnvm1","/dev/libnvm2","/dev/libnvm3","/dev/libnvm4","/dev/libnvm5"};
  std::vector<Controller *> ctrls;

  uint32_t n_ctrls = 1;
  uint64_t queueDepth = 1024;
  uint64_t numQueues = 128;
  
  uint32_t cudaDevice = 0;
  uint32_t nvmNamespace = 1;
  
  //member functions
  void init_GIDS_controllers(uint32_t num_ctrls, uint64_t q_depth, uint64_t num_q,  const std::vector<int>& ssd_list, uint32_t device, bool sim);

};

template <typename TYPE>
struct GIDS_CPU_buffer {
    TYPE* cpu_buffer;
    TYPE* device_cpu_buffer;
    uint64_t cpu_buffer_dim;
    uint64_t cpu_buffer_len;
};





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

    uint32_t gpu_ways = 32;
    uint32_t cpu_ways = 32;

    uint32_t cudaDevice = 0;

    size_t blkSize = 128;
 
    
    std::vector<Controller *> ctrls;


    int rank = 0;
    int nranks = 1;
    int mype_node = 0;

    unsigned int* d_request_counters;

    cudaStream_t stream;
 
    float kernel_time = 0; 
    float request_time = 0;
    float request_kernel_time = 0;
    uint64_t total_access = 0;

    bool simulation = false;
    void* sim_buf = nullptr;

    void init_cache(Dist_GIDS_Controllers GIDS_ctrl, uint32_t ps, uint64_t read_off, uint64_t gpu_cache_size, uint64_t cpu_cache_size, uint32_t num_gpus, uint64_t num_ele, uint64_t num_ssd, uint64_t ways, bool is_simulation, const std::string &feat_file, int off);
    void read_feature(uint64_t i_ptr, uint64_t i_index_ptr, int64_t num_index, int dim, int cache_dim);
    void dist_read_feature(uint64_t i_return_tensor_ptr, uint64_t i_nvshmem_index_ptr, int64_t max_index, int dim, int cache_dim);

    void print_stats();
    void update_NVshmem_metadata(int my_rank, int num_ranks, int pe_node);
    void send_requests(uint64_t i_src_index_ptr, int64_t num_index, uint64_t i_nvshmem_request_ptr, int max_index);


    void init();
    uint64_t allocate(int);
    void free(uint64_t);
    void finalize();
    int get_rank();
    int get_world_size();
    int get_mype_node();
    void NVshmem_quiet();
    ~NVSHMEM_Cache();
    //MPI functions

    
};

struct Emulate_Cache {
  Emulate_SA_handle<float>* Emul_SA_handle;
  Emulate_SA_cache_d_t<float>* Emul_cache_ptr;

  void init_cache(uint64_t num_sets, uint64_t num_ways, uint64_t page_size, uint32_t cudaDevice, uint8_t eviction_policy, int, uint64_t);
  void read_feature(uint64_t, uint64_t,int64_t , int , int , uint64_t , uint64_t );
  void read_feature_with_color(uint64_t, uint64_t,int64_t , int , int , uint64_t , uint64_t, uint64_t);
  void distribute_node(uint64_t items, uint64_t score_ten_ptr, uint64_t dist_ten_ptr, uint64_t counter_ten_ptr, uint64_t dict_ptr, uint64_t color_ptr, int items_len, int num_nvlink, uint64_t num_colors);
  void print_counters();
  void read_data(uint64_t i_ptr, int n);
  void write_data(uint64_t i_ptr, int n);

  float color_score(uint64_t);
  void copy_meta(uint64_t i_dst_ptr);
  void distribute_node_with_cache_meta(uint64_t i_item_ptr, uint64_t i_color_tensor_ptr, const std::vector<uint64_t>& return_ptr_list, const std::vector<uint64_t>&  meta_data_list, int tensor_size, int num_parts);
  void distribute_node_with_affinity(uint64_t i_item_ptr,uint64_t i_color_tensor_ptr, const std::vector<uint64_t>& return_ptr_list, const std::vector<uint64_t>&  meta_data_list, uint64_t i_topk_ptr,  uint64_t i_score_ptr, int topk, int tensor_size, int num_parts);

  ~Emulate_Cache();
};



#endif
