#ifndef BAMNVME_H
#define BAMNVME_H

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

#include "set_associative_page_cache.h"


//#define TYPE float
struct GIDS_Controllers {
  const char *const ctrls_paths[6] = {"/dev/libnvm0","/dev/libnvm1","/dev/libnvm2","/dev/libnvm3","/dev/libnvm4","/dev/libnvm5"};
  std::vector<Controller *> ctrls;

  uint32_t n_ctrls = 1;
  uint64_t queueDepth = 1024;
  uint64_t numQueues = 128;
  
  uint32_t cudaDevice = 0;
  uint32_t nvmNamespace = 1;
  
  //member functions
  void init_GIDS_controllers(uint32_t num_ctrls, uint64_t q_depth, uint64_t num_q,  const std::vector<int>& ssd_list, uint32_t device);

};

template <typename TYPE>
struct GIDS_CPU_buffer {
    TYPE* cpu_buffer;
    TYPE* device_cpu_buffer;
    uint64_t cpu_buffer_dim;
    uint64_t cpu_buffer_len;
};


template <typename TYPE>
struct BAM_Feature_Store {


  GIDS_CPU_buffer<TYPE> CPU_buffer;
  
  bool set_associative_cache = false;
  GIDS_SA_handle<TYPE>* SA_handle;
  SA_cache_d_t<TYPE>* cache_ptr;

  //Dist GIDS
  uint64_t** sampling_node_counter_list;
  uint64_t* meta_buffer[8];  
  uint64_t** d_meta_buffer;

  //GIDS optimization flasg
  bool cpu_buffer_flag = false;
  bool seq_flag = true;
  //Sampling Offsets
  uint64_t* offset_array;

  int dim;
  uint64_t total_access;
  unsigned int cpu_access_count = 0;
  unsigned int* d_cpu_access;

  //BAM parameters
  uint32_t cudaDevice = 0;
  size_t numPages = 262144 * 8;
  bool stats = false;
  size_t numThreads = 64;
  uint32_t domain = 0;
  uint32_t bus = 0;
  uint32_t devfn = 0;

  uint32_t n_gpus = 1;
  uint32_t n_ctrls = 1;
  size_t blkSize = 128;
  size_t queueDepth = 1024;
  size_t numQueues = 128;
  uint32_t pageSize = 4096 ;
  uint64_t numElems = 300LL*1000*1000*1024;
  uint64_t read_offset = 0;
  std::vector<Controller *> ctrls;

  page_cache_t *h_pc;
  range_t<TYPE> *h_range;
  std::vector<range_t<TYPE> *> vr;
  array_t<TYPE> *a;
  range_d_t<TYPE> *d_range;

  //wb
  uint32_t wb_size = 0;
  uint32_t pvp_depth = 0;

  //PVP
  bool use_PVP = false;
  TYPE* PVP_pinned_data;
  uint64_t* PVP_pinned_idx;
  int32_t head_ptr = 0 ;
  uint64_t num_evicted_cl = 0;

  uint64_t** node_flag_buffer;
  uint64_t** node_flag_buffer_array ;
  uint64_t max_sample_size;


  uint8_t update_counter = 0;
  uint8_t refresh_time = 1;


  //Debugging 
  bool debug_mode = false;
  unsigned long long * evict_counter;
  unsigned long long* prefetch_counter;

  //Eviciton Policy
  uint8_t eviction_policy;
  //Pinned memory pointer in host/device
  uint8_t* h_static_val_array;
  uint8_t* d_static_val_array;


  int low_priority;
  int high_priority;
  cudaStream_t transfer_stream;
  cudaStream_t fill_stream;
  cudaStream_t update_stream;


  float kernel_time = 0; 
  bool first = true;



  void init_controllers(GIDS_Controllers GIDS_ctrl, uint32_t ps, uint64_t r_off, uint64_t num_ele, uint64_t cache_size, 
                        uint64_t num_ssd);
  void init_set_associative_cache(GIDS_Controllers , uint32_t , uint64_t , uint64_t , uint32_t ,  uint64_t , uint64_t , uint64_t , bool , bool , uint32_t , uint32_t , uint64_t , uint8_t, int , int );

  

  void read_feature(uint64_t tensor_ptr, uint64_t index_ptr,int64_t num_index, int dim, int cache_dim, uint64_t key_off);
  void read_feature_hetero(int num_iter, const std::vector<uint64_t>&  i_ptr_list, const std::vector<uint64_t>& i_index_ptr_list, const std::vector<uint64_t>&   num_index, int dim, int cache_dim, const std::vector<uint64_t>& key_off);
  void read_feature_merged(int num_iter, const std::vector<uint64_t>&  i_ptr_list, const std::vector<uint64_t>& i_index_ptr_list, const std::vector<uint64_t>&   num_index, int dim, int cache_dim);
  void read_feature_merged_hetero(int num_iter, const std::vector<uint64_t>&  i_ptr_list, const std::vector<uint64_t>& i_index_ptr_list, const std::vector<uint64_t>&   num_index, int dim, int cache_dim, const std::vector<uint64_t>& key_off);


  void SA_read_feature(uint64_t tensor_ptr, uint64_t index_ptr,int64_t num_index, int dim, int cache_dim, uint64_t key_off, uint64_t staic_info_ptr);
  
  void SA_read_feature_dist( const std::vector<uint64_t>&  i_return_ptr_list, const std::vector<uint64_t>&  i_index_ptr_list, const std::vector<uint64_t>&  index_size_list, int num_gpu, int dim, int cache_dim, uint64_t key_off, const std::vector<uint64_t>&  i_static_info_ptr_list);
  void cpu_backing_buffer(uint64_t dim, uint64_t len);
  void set_cpu_buffer(uint64_t idx_buffer, int num);  

  void set_window_buffering(uint64_t id_idx,  int64_t num_pages, int hash_off); 
  void print_stats();
  void print_stats_no_ctrl();

 
  uint64_t get_array_ptr();
  uint64_t get_offset_array();
  void set_offsets(uint64_t in_off, uint64_t index_off, uint64_t data_off);
  void store_tensor(uint64_t tensor_ptr, uint64_t num, uint64_t offset);
  void read_tensor( uint64_t num, uint64_t offset);
  void flush_cache();
  unsigned int get_cpu_access_count();
  void flush_cpu_access_count();

  void split_node_list_init(uint64_t i_index_ptr, int64_t num_gpu, int64_t index_size, uint64_t i_index_pointer_list);
  void split_node_list(uint64_t i_index_ptr, int64_t num_gpu, int64_t index_size, uint64_t i_bucket_ptr_list, uint64_t i_index_pointer_list, uint64_t i_meta_buffer);
  void reset_node_counter(const std::vector<uint64_t>&  i_index_pointer_list, int num_gpu);
  void create_meta_buffer(uint64_t num_gpu, uint64_t max_size);
  void gather_feature_list(uint64_t i_return_ptr, const std::vector<uint64_t>&  i_return_ptr_list, const std::vector<uint64_t>&  index_size_list, 
                                            int num_gpu, int dim, int my_rank, const std::vector<uint64_t>&  i_meta_buffer) ;
  
  void print_meta_buffer(const std::vector<uint64_t>&  index_size_list, int num_gpu , int rank);

  //PVP
  void prefetch_from_victim_queue();
  void fill_batch();

  void print_victim_buffer_index(uint64_t offset, uint64_t len);
  void print_victim_buffer_data(uint64_t offset, uint64_t len);

  void update_reuse_counters(uint64_t batch_array_idx, uint64_t batch_size_idx, uint32_t max_batch_size, int num_gpus, int num_buffers);

  //Eviciton Poliyc
  void get_static_info(uint64_t i_out, uint64_t i_index_ptr, uint64_t index_len);
  void get_static_info_dist(const std::vector<uint64_t>&  i_out, const std::vector<uint64_t>&  i_index_ptr, const std::vector<uint64_t>&  index_len_array);
  void create_static_info_buffer(const std::string& path);

};


#endif
