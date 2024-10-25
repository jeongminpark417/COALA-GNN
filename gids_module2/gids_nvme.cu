#ifndef __GIDS_NVME_CU__
#define __GIDS_NVME_CU__




#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <tuple>

#include <stdio.h>
#include <vector>
#include <regex>

#include <bam_nvme.h>
#include <pybind11/stl.h>

//#include "gids_kernel.cu"
#include "nvshmem_cache_kernel.cu"
#include "emual_kernel.cu"


typedef std::chrono::high_resolution_clock Clock;

void Dist_GIDS_Controllers::init_GIDS_controllers(uint32_t num_ctrls, uint64_t q_depth, uint64_t num_q, 
                          const std::vector<int>& ssd_list, uint32_t device, bool simulation){

  n_ctrls = num_ctrls;
  queueDepth = q_depth;
  numQueues = num_q;
  cudaDevice = device;
  cudaSetDevice(cudaDevice);

}


// Function to load file content into a memory buffer using malloc
char* load_file_to_memory(const std::string& file_path, size_t& file_size) {
    std::ifstream file(file_path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Unable to open file: " + file_path);
    }

    // Get the size of the file
    file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Allocate buffer memory using malloc
    char* buffer;
    cuda_err_chk(cudaHostAlloc((void**)&buffer, file_size, cudaHostAllocMapped));
    if (buffer == nullptr) {
        throw std::runtime_error("Failed to allocate memory for file buffer.");
    }

    // Read file into the buffer
    if (!file.read(buffer, file_size)) {
        free(buffer);  // Free the buffer if reading fails
        throw std::runtime_error("Error reading the file: " + file_path);
    }

    return buffer;
}


void  parse_numpy_file(const std::string &color_file, int dim, char*& ret_buffer, int64_t*& ret_buffer_ptr, std::vector<int>& shape){
  size_t file_size;
  
  char* buffer = load_file_to_memory(color_file, file_size);
  char* buffer_ptr = buffer;
  std::string magic_string(buffer_ptr, 6);
    if (magic_string != "\x93NUMPY") {
        throw std::runtime_error("Not a valid .npy file.");
    }

  uint8_t major_version = buffer_ptr[6];
  uint8_t minor_version = buffer_ptr[7];
  std::cout << "Version: " << static_cast<int>(major_version) << "." << static_cast<int>(minor_version) << std::endl;
  buffer_ptr += 8; // Move past the magic string and version bytes

  // Step 3: Read the header length based on the version
  uint32_t header_len = 0;
  if (major_version == 1) {
      // Version 1.x uses 2 bytes for header length
      uint16_t header_len_16;
      memcpy(&header_len_16, buffer_ptr, sizeof(uint16_t));
      header_len = header_len_16;
      buffer_ptr += sizeof(uint16_t);  // Move past the header length
  } else if (major_version == 2) {
      // Version 2.x uses 4 bytes for header length
      memcpy(&header_len, buffer_ptr, sizeof(uint32_t));
      buffer_ptr += sizeof(uint32_t);  // Move past the header length
  } else {
      throw std::runtime_error("Unsupported .npy file version: " + std::to_string(major_version));
  }


  std::string header(buffer_ptr, header_len);
  buffer_ptr += header_len;  // Move the pointer past the header

  // Step 5: Parse the header using regex to extract shape, dtype, and fortran order
  std::regex shape_regex_1D(R"('shape':\s?\((\d+),?\))");
  std::regex shape_regex_2D(R"('shape':\s?\((\d+),\s?(\d+)\))");
  std::regex dtype_regex(R"('descr':\s*'(.*?)')");
  std::regex fortran_regex(R"('fortran_order':\s?(True|False))");

  std::smatch match;

  // Extract shape
  if(dim == 1){
    if (std::regex_search(header, match, shape_regex_1D)) {
        int shape_value = std::stoi(match[1].str());
        shape.push_back(shape_value);
    }
  }
  else if(dim == 2){
    if (std::regex_search(header, match, shape_regex_2D)) {
        int shape_value1 = std::stoi(match[1].str());
        int shape_value2 = std::stoi(match[2].str());
        shape.push_back(shape_value1);
        shape.push_back(shape_value2);
    }
  }
  else{
    throw std::runtime_error("Unsupported dimension for .npy file\n");
  }

  if (std::regex_search(header, match, dtype_regex)) {
      auto dtype= match[1].str();
      assert(dtype == "<i8" && "The dtype is not '<i8'. This program only supports 64-bit integers.");
  }

  ret_buffer = buffer;
  ret_buffer_ptr = (int64_t*) buffer_ptr;

  return;

}

template <typename TYPE>
void NVSHMEM_Cache<TYPE>::init_cache(Dist_GIDS_Controllers GIDS_ctrl, uint32_t ps, uint64_t read_off, uint64_t gpu_cache_size, uint64_t cpu_cache_size, uint32_t num_gpus, uint64_t num_ele, uint64_t num_ssd, uint64_t n_ways, bool is_simulation,
                                      const std::string &feat_file, int file_off,
                                      bool use_color_data, const std::string &color_file, const std::string &topk_file
                                     ) {

  num_eles = num_ele;
  read_offset = read_off;
  n_ctrls = num_ssd;
  n_gpus = num_gpus;

  //num_ways = gpu_ways + cpu_ways;
  num_ways = n_ways;

  page_size = ps;
  dim = ps / sizeof(TYPE);

  ctrls = GIDS_ctrl.ctrls;
  cudaDevice = GIDS_ctrl.cudaDevice;
  
  simulation = is_simulation;


  cudaSetDevice(cudaDevice);

  gpu_ways = (gpu_cache_size * n_ways) / (gpu_cache_size + cpu_cache_size);
  cpu_ways = num_ways - gpu_ways;

  n_pages = (gpu_cache_size + cpu_cache_size) * 1024LL*1024/page_size;


  std::cout << "n pages: " << (int)(this->n_pages) <<std::endl;
  std::cout << "page size: " << (int)(this->page_size) << std::endl;
  std::cout << "num elements: " << this->num_eles << std::endl;
  std::cout << "cudaDevice" << cudaDevice  << std::endl;
  std::cout << "number of GPUs: " << n_gpus  << std::endl;


  std::cout << "CPU ways: " << cpu_ways  << std::endl;
  std::cout << "GPU ways: " << gpu_ways  << std::endl;
  std::cout << "Total ways: " << num_ways  << std::endl;
  std::cout << "Simulation: " << is_simulation << std::endl;

/*
  if(is_simulation){
    std::ifstream file(feat_file, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << feat_file << std::endl;
        return ;
    }

    std::cout << "File: " << feat_file << " is opend\n";
    // Get the size of the file
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::streamsize remaining_size = file_size - file_off;

    // Move the file pointer to the N-th byte
    file.seekg(file_off, std::ios::beg);

    // Create a string to hold the file contents after N bytes
    std::string file_data(remaining_size, '\0');

    // Read the file starting from the N-th byte
    if (!file.read(&file_data[0], remaining_size)) {
        std::cerr << "Failed to read file: " << feat_file << std::endl;
        return ;
    }
    cuda_err_chk(cudaHostAlloc(&sim_buf, remaining_size, cudaHostAllocDefault));
  
    // Copy the file data from the string to the pinned memory
    std::memcpy(sim_buf, file_data.data(), remaining_size);
    
    std::cout << "Simulation Buffer created\n";
  }
*/
  

  if(use_color_data){
    std::vector<int> color_shape;
    parse_numpy_file(color_file, 1, color_ptr, color_buffer_ptr, color_shape);


    parse_numpy_file(topk_file, 2, topk_ptr, topk_buffer_ptr, topk_shape);
    num_colors = topk_shape[0];
    printf("\t\t use color data num_colors: %i\n", num_colors);

  }

  num_sets = n_pages / num_ways;
  cache_handle = new NVSHMEM_cache_handle<TYPE>(num_sets, gpu_ways, page_size, /*ctrls[0][0],*/ ctrls, cudaDevice, cudaDevice, num_gpus, cpu_ways, is_simulation, sim_buf,
                                                use_color_data, num_colors, color_buffer_ptr);
  cache_ptr = cache_handle -> get_ptr();

  cuda_err_chk(cudaDeviceSynchronize());
  cuda_err_chk(cudaMalloc(&d_request_counters, sizeof(unsigned int) * n_gpus));

  printf("Init done\n");

  return;
}


template <typename TYPE>
void NVSHMEM_Cache<TYPE>::get_cache_data(int64_t ret_i_ptr){
  int32_t* cache_meta_data = cache_handle->get_color_counter_ptr();
  int32_t* dst = (int32_t*) ret_i_ptr;

  int32_t sum = 0;
  for(int i = 0; i < num_colors; i++ ){
    dst[i] = cache_meta_data[i];
  }
 // printf("sum:%i\n", sum);
  return;
  // return py::array_t<int32_t>(
  //       {num_colors},                // Shape of the array
  //       {sizeof(int32_t)},     // Strides (in bytes)
  //       cache_meta_data                   // Pointer to the data, no capsule for freeing memory
  //   );
}

template <typename TYPE>
int NVSHMEM_Cache<TYPE>::get_num_colors(){

  return num_colors;
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

    if(simulation == false){
      for(int i = 0; i < n_ctrls; i++){
      std::cout << "print ctrl reset " << i << ": ";
        (this->ctrls[i])->print_reset_stats();
        std::cout << std::endl;
      }
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


void  gather_local_ranks(MPI_Comm node_comm, int system_rank, int node_world_size, int* local_ranks) {
    // Vector to hold the gathered ranks from all processes in the node communicator
    std::vector<int> temp_local_ranks(node_world_size);

    // Perform an allgather operation to collect system_rank from all processes in the node_comm communicator
    MPI_Allgather(&system_rank, 1, MPI_INT, temp_local_ranks.data(), 1, MPI_INT, node_comm);

    // Print out the gathered local ranks
    std::cout << "Local ranks within the node: ";
    int i = 0;
    for (int rank : temp_local_ranks) {
        std::cout << rank << " ";
        local_ranks[i] = rank;
        i++;

    }
    
}

void gather_master_ranks( int system_rank,  int system_world_size, int node_rank,  std::vector<int>& master_ranks){
  std::vector<int> gathered_values(system_world_size);
  int send_value = (node_rank == 0) ? system_rank : -1;
  MPI_Allgather(&send_value, 1, MPI_INT, gathered_values.data(), 1, MPI_INT, MPI_COMM_WORLD);

  std::cout << "Master ranks: ";
  for (int rank : gathered_values) {
    if(rank != -1){
      std::cout << rank << " ";
      master_ranks.push_back(rank);
    }
  }
}




  template <typename TYPE>
  void NVSHMEM_Cache<TYPE>::init_with_arg(py::list argv) {
    //nvshmem_init();
    
   // nvshmemx_init_attr_t attr;
   // MPI_Init(&argc, &argv);
   //if(is_torch_distributed_used == 0)
    int argc = argv.size();  // Size of argv list gives argc

    // Convert py::list to std::vector<std::string> for easier C++ use
    std::vector<std::string> args = argv.cast<std::vector<std::string>>();
    
    char** c_argv = (char**) malloc(sizeof(char*) * argc);
        int i = 0;
        for (auto& arg : args) {
            //c_argv.push_back(const_cast<char*>(arg.c_str()));  // Get C-style string
            c_argv[i] = const_cast<char*>(arg.c_str());
            i++;
        }

    MPI_Init(&argc, &c_argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_nrank);
  //  MPI_Comm mpi_comm = MPI_COMM_TYPE_SHARED;
    //MPI_Comm mpi_comm = MPI_COMM_TYPE_SHARED;


   
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
    MPI_Comm_size(node_comm, &node_nrank);
    MPI_Comm_rank(node_comm, &node_rank);

    cuda_err_chk(cudaSetDevice(node_rank));

    local_ranks = new int[node_nrank];
    gather_local_ranks(node_comm, mpi_rank, node_nrank, local_ranks);
    MPI_Barrier(node_comm);

    gather_master_ranks( mpi_rank, mpi_nrank, node_rank, master_ranks);
    MPI_Barrier(MPI_COMM_WORLD);

    attr.mpi_comm = &node_comm;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    
    rank = nvshmem_my_pe();
    nranks = nvshmem_n_pes();
    MPI_Barrier(node_comm);

    printf("My PE %d rank:%d World size: %d node rank: %d NVSHMEM initialized and CUDA device set to %d\n", mype_node, mpi_rank, mpi_nrank, node_rank, mype_node);


    
//    free((void*)c_argv);
    printf("init done\n");
    

  }


  template <typename TYPE>
  void NVSHMEM_Cache<TYPE>:: init(int is_torch_distributed_used) {
    //nvshmem_init();
    
   // nvshmemx_init_attr_t attr;
   // MPI_Init(&argc, &argv);
   //if(is_torch_distributed_used == 0)
    
    MPI_Init(NULL, NULL);

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_nrank);
  //  MPI_Comm mpi_comm = MPI_COMM_TYPE_SHARED;
    //MPI_Comm mpi_comm = MPI_COMM_TYPE_SHARED;


   
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
    MPI_Comm_size(node_comm, &node_nrank);
    MPI_Comm_rank(node_comm, &node_rank);

    cuda_err_chk(cudaSetDevice(node_rank));

    local_ranks = new int[node_nrank];
    gather_local_ranks(node_comm, mpi_rank, node_nrank, local_ranks);
    MPI_Barrier(node_comm);

    gather_master_ranks( mpi_rank, mpi_nrank, node_rank, master_ranks);
    MPI_Barrier(MPI_COMM_WORLD);

    attr.mpi_comm = &node_comm;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    
    rank = nvshmem_my_pe();
    nranks = nvshmem_n_pes();
    MPI_Barrier(node_comm);

    printf("My PE %d rank:%d World size: %d node rank: %d NVSHMEM initialized and CUDA device set to %d\n", mype_node, mpi_rank, mpi_nrank, node_rank, mype_node);


    

    printf("init done\n");
    

  }

  template <typename TYPE>
  std::vector<int> NVSHMEM_Cache<TYPE>::get_local_ranks() {
    std::vector<int>  ret(node_nrank);
    for (int i = 0; i < node_nrank; i++){
      ret[i] = local_ranks[i];
    }
    return ret;
  }

  template <typename TYPE>
  std::vector<int> NVSHMEM_Cache<TYPE>::get_master_ranks() {
    return master_ranks;
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
  int NVSHMEM_Cache<TYPE>:: MPI_get_rank() {
      return mpi_rank;
  }
  template <typename TYPE>
  int NVSHMEM_Cache<TYPE>:: MPI_get_world_size() {
      return mpi_nrank;
  }

  template <typename TYPE>
  int NVSHMEM_Cache<TYPE>:: node_get_rank() {
      return node_rank;
  }
  template <typename TYPE>
  int NVSHMEM_Cache<TYPE>:: node_get_world_size() {
      return node_nrank;
  }


  template <typename TYPE>
  int NVSHMEM_Cache<TYPE>:: get_mype_node(){
      return mype_node;
  }

  template <typename TYPE>
  void NVSHMEM_Cache<TYPE>:: NVshmem_quiet(){
      nvshmem_quiet();
  }

  template <typename TYPE>
  NVSHMEM_Cache<TYPE>::~NVSHMEM_Cache() {
      if(sim_buf != nullptr)
          cudaFreeHost (sim_buf);

      if(color_ptr != nullptr)
        cudaFreeHost (color_ptr);

      delete[] local_ranks;
      MPI_Finalize();
  }



void Emulate_Cache::read_feature(uint64_t i_ptr, uint64_t i_index_ptr,
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

void Emulate_Cache::read_feature_with_color(uint64_t i_ptr, uint64_t i_index_ptr,
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


void Emulate_Cache::init_cache(uint64_t num_sets, uint64_t num_ways, uint64_t page_size, uint32_t cudaDevice, uint8_t eviction_policy, int cache_track, uint64_t num_colors){
    bool c_track = false;
    if(cache_track == 1)
      c_track = true;
    printf("Init Cache number of sets: %lli num_ways: %lli\n", num_sets, num_ways);
	  Emul_SA_handle = new Emulate_SA_handle<float>(num_sets, num_ways, page_size, cudaDevice, eviction_policy, c_track, num_colors);
	  Emul_cache_ptr = Emul_SA_handle -> get_ptr();
}

void Emulate_Cache::print_counters(){
  printf("Printing Cache Counters\n");
  //Emul_SA_handle -> print_counters();
  Emulate_SA_print_counters<<<1,1>>>(Emul_cache_ptr);
}

float Emulate_Cache::color_score(uint64_t color){
  return Emul_SA_handle -> color_score(color);
}

void Emulate_Cache::distribute_node(uint64_t items, uint64_t score_ten_ptr, uint64_t dist_ten_ptr, uint64_t counter_ten_ptr, uint64_t dict_ptr, uint64_t color_ptr, int items_len, int num_nvlink, uint64_t num_colors){
  //return Emul_SA_handle -> color_score(color);
    cuda_err_chk(cudaDeviceSynchronize());

    uint64_t b_size = 64;
    uint64_t g_size = (items_len + b_size - 1) / b_size; 

//    dim3 block_dim = (b_size, num_nvlink, 1);

    float *score_ptr = (float *)score_ten_ptr;
	  int64_t *index_ptr = (int64_t *)dist_ten_ptr;
	  int* counter_ptr = (int*) counter_ten_ptr;
	  int64_t *items_ptr = (int64_t *)items;

    float* dict = (float*) dict_ptr;
    int64_t* color_p = (int64_t*) color_ptr;

  int shared_memory_size = b_size * num_nvlink * sizeof(float);

  //self.Emul_SA.distribute_node(score_ten.data_ptr(), dist_list_ten.data_ptr(), counter_ten.data_ptr(), len(items), self.cache_track.num_gpus)
    distribute_node_kernel<<<g_size, b_size, shared_memory_size>>>(items_ptr, score_ptr, index_ptr, counter_ptr, dict, color_p, items_len, num_nvlink, num_colors);
      cuda_err_chk(cudaDeviceSynchronize());

  return;
}

  __global__
  void read_k(float* ptr, int n){
  if(threadIdx.x == 0 && blockIdx.x ==0){
    for(int i = 0; i < n; i++){
      printf("IDX:%i val:%f\n", n, ptr[i]);
    }
  }

  }

  void Emulate_Cache::read_data(uint64_t i_ptr, int n){
  float* ptr = (float*) i_ptr;
  read_k<<<1,1>>>(ptr, n);
  cuda_err_chk(cudaDeviceSynchronize());

  }

  __global__
  void write_k(float* ptr, int n){
  if(threadIdx.x == 0 && blockIdx.x ==0){
    for(int i = 0; i < n; i++){
      ptr[i] = i + 100.0;
    }
  }

  }

  void Emulate_Cache::write_data(uint64_t i_ptr, int n){
  float* ptr = (float*) i_ptr;
  write_k<<<1,1>>>(ptr, n);
  cuda_err_chk(cudaDeviceSynchronize());

  }


  void Emulate_Cache::distribute_node_with_cache_meta(uint64_t i_item_ptr,uint64_t i_color_tensor_ptr, const std::vector<uint64_t>& return_ptr_list, const std::vector<uint64_t>&  meta_data_list, int tensor_size, int num_parts){
  
   // printf("start distribute node with cache meta\n");
    int64_t* item_ptr = (int64_t*)i_item_ptr;
    int64_t* color_tensor_ptr = (int64_t*)i_color_tensor_ptr;
    int item_len = tensor_size * num_parts;
    
    int bucket_len[num_parts] =  {0};  
    double max_score = -1.0;
    int cur_max_part = 0;

    for(int i = 0; i < item_len; i++){
      int64_t node_id = item_ptr[i];
     // printf("node id: %lli\n", node_id);
      int64_t node_color = color_tensor_ptr[i];
      cur_max_part = 0;
      max_score = -1.0;
      for(int j = 0; j < num_parts; j++){
        double cur_score = (double)(((int32_t*)(meta_data_list[j]))[node_color]);
        if(node_color == 0){
          cur_score = 0.0;
        } 
        if(cur_score <= -1.0){
          printf("node color %lli score %d\n", node_color, cur_score);
        }


        if(bucket_len[j] == tensor_size)
          cur_score = -1.0;
        
       //printf("id: %i score: %d\n", j, cur_score);
        if(cur_score > max_score)
          cur_max_part = j;
      }
     // printf("write at %i idx: %i\n", cur_max_part,bucket_len[cur_max_part] );
      ((int64_t*)(return_ptr_list[cur_max_part]))[bucket_len[cur_max_part]] = node_id;
      bucket_len[cur_max_part]+=1;
    }
  }


  void Emulate_Cache::distribute_node_with_affinity(uint64_t i_item_ptr,uint64_t i_color_tensor_ptr, const std::vector<uint64_t>& return_ptr_list, const std::vector<uint64_t>&  meta_data_list, uint64_t i_topk_ptr,  uint64_t i_score_ptr, int topk, int tensor_size, int num_parts){
  
   // printf("start distribute node with cache meta\n");
    int64_t* item_ptr = (int64_t*) i_item_ptr;
    int64_t* topk_ptr = (int64_t*) i_topk_ptr;
    double* score_ptr = (double*) i_score_ptr;

    int64_t* color_tensor_ptr = (int64_t*)i_color_tensor_ptr;
    int item_len = tensor_size * num_parts;
    
    int bucket_len[num_parts] =  {0};  
    double max_score = -1.0;
    int cur_max_part = 0;

    for(int i = 0; i < item_len; i++){
      int64_t node_id = item_ptr[i];
     // printf("node id: %lli\n", node_id);
      int64_t node_color = color_tensor_ptr[i];
      cur_max_part = 0;
      max_score = -1.0;
      for(int j = 0; j < num_parts; j++){
        
        double cur_score = (double)(((int32_t*)(meta_data_list[j]))[node_color]);
        if(node_color == 0){
          cur_score = 0.0;
        } 
        else{
          for(int k = 0; k < topk; k++){
            int64_t neigh_color = topk_ptr[(node_color - 1) * topk + k];
            double neigh_affinity = score_ptr[(node_color - 1) * topk + k];
            double neigh_score = 0.0;
            int32_t meta_data = 0;
            if(neigh_color != 0){
              neigh_score = (double)(((int32_t*)(meta_data_list[j]))[neigh_color]);
              meta_data = (((int32_t*)(meta_data_list[j]))[neigh_color]);
              cur_score += (neigh_score * neigh_affinity * 5);
            }
            if(cur_score <= -1.0){
              printf("node color %lli neigh color: %lli score %f neigh_affinity: %f neigh score: %f meta data:%li idx:%i\n", node_color, neigh_color, cur_score, neigh_affinity, neigh_score, meta_data, j);
            }
          }
        }
        
        // if(cur_score <= -1.0){
        //   printf("node color %lli score %f neigh_affinity: %f neigh score: %f\n", node_color, cur_score, neigh_affinity, neigh_score);
        // }


        if(bucket_len[j] == tensor_size)
          cur_score = -1.0;
        
       //printf("id: %i score: %d\n", j, cur_score);
        if(cur_score > max_score)
          cur_max_part = j;
      }
     // printf("write at %i idx: %i\n", cur_max_part,bucket_len[cur_max_part] );
      ((int64_t*)(return_ptr_list[cur_max_part]))[bucket_len[cur_max_part]] = node_id;
      bucket_len[cur_max_part]+=1;
    }
  }



  void Emulate_Cache::copy_meta(uint64_t i_dst_ptr){
    int32_t* dst_ptr = (int32_t*) i_dst_ptr;
    int32_t* src_ptr =  Emul_SA_handle -> get_meta_ptr();

    memcpy(dst_ptr, src_ptr, sizeof(int32_t) * (Emul_SA_handle -> num_color_));
  }
  
  Emulate_Cache::~Emulate_Cache() {
      if (Emul_SA_handle) {
          delete Emul_SA_handle;  // Free the dynamically allocated Emulate_SA_handle
          Emul_SA_handle = nullptr;  // Set pointer to null to avoid dangling pointers
      }
  }



PYBIND11_MODULE(Dist_Cache, m) {
  m.doc() = "Python bindings for an example library";

  namespace py = pybind11;

      py::class_<Dist_GIDS_Controllers>(m, "Dist_GIDS_Controllers")
      .def(py::init<>())
      .def("init_GIDS_controllers", &Dist_GIDS_Controllers::init_GIDS_controllers);


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
      .def("NVshmem_quiet", &NVSHMEM_Cache<float>::NVshmem_quiet)
      .def("MPI_get_rank", &NVSHMEM_Cache<float>::MPI_get_rank)
      .def("MPI_get_world_size", &NVSHMEM_Cache<float>::MPI_get_world_size)
      .def("node_get_rank", &NVSHMEM_Cache<float>::node_get_rank)
      .def("node_get_world_size", &NVSHMEM_Cache<float>::node_get_world_size)
      .def("get_local_ranks", &NVSHMEM_Cache<float>::get_local_ranks)
      .def("get_master_ranks", &NVSHMEM_Cache<float>::get_master_ranks)

      .def("get_cache_data", &NVSHMEM_Cache<float>::get_cache_data)
      .def("get_num_colors", &NVSHMEM_Cache<float>::get_num_colors)

        .def("init_with_arg", &NVSHMEM_Cache<float>::init_with_arg)


      ;

      py::class_<Emulate_Cache>(m, "Emulate_Cache")
      .def(py::init<>())
      .def("init_cache", &Emulate_Cache::init_cache)
      .def("print_counters", &Emulate_Cache::print_counters)
      .def("color_score", &Emulate_Cache::color_score)
 			.def("read_feature", &Emulate_Cache::read_feature)
 			.def("read_data", &Emulate_Cache::read_data)
 			.def("write_data", &Emulate_Cache::write_data)
 			.def("distribute_node", &Emulate_Cache::distribute_node)
 			.def("distribute_node_with_cache_meta", &Emulate_Cache::distribute_node_with_cache_meta)
      	.def("distribute_node_with_affinity", &Emulate_Cache::distribute_node_with_affinity)

      .def("copy_meta", &Emulate_Cache::copy_meta)

      .def("read_feature_with_color", &Emulate_Cache::read_feature_with_color);


;

}



//gids



#endif

