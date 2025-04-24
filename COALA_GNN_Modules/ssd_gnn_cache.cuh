#pragma once
#include <stdexcept>  // for std::runtime_error
#include <ctrl.h>  // for NVME controller
#include "nvshmem_cache.h"
#include "isolated_cache.h"
#include "cache_kernel.cu"
#include "node_distributor_pybind.cuh"

//Datatype for feature information is Float
class SSD_GNN_SSD_Controllers{
    public:
        uint64_t queueDepth = 1024;
        uint64_t numQueues = 128;
        
        uint32_t n_ctrls;
        uint32_t cudaDevice;
        uint64_t offset;
        uint64_t num_elements;

        uint32_t nvmNamespace = 1;
        uint32_t page_size = 4096;
        int dim;
        int cache_dim;

        bool SSD_SIM = true;
        const char *const ctrls_paths[6] = {"/dev/libnvm0","/dev/libnvm1","/dev/libnvm2","/dev/libnvm3","/dev/libnvm4","/dev/libnvm5"};
        std::vector<Controller *> ctrls;


    SSD_GNN_SSD_Controllers(uint32_t num_ctrls, uint32_t p_size, uint64_t n_elems, uint64_t read_off, uint32_t device_id, int feat_dim, bool sim)
        : n_ctrls(num_ctrls), page_size(p_size), num_elements(n_elems), cudaDevice(device_id), offset(read_off), dim(feat_dim), SSD_SIM(sim){
        
        
        if(dim <= 128)
            cache_dim = 128;
        else if(dim <= 256)
            cache_dim = 256;
        else if(dim <= 512)
            cache_dim = 512;
        else if(dim <= 1024)
            cache_dim = 1024;
    
        else
            throw std::runtime_error("Only Feature Embedding Size less than 8KB is supported\n");

        cudaSetDevice(cudaDevice);
        page_size = cache_dim * sizeof(float);

        if(!SSD_SIM){
            for (size_t i = 0; i < n_ctrls; i++) {
            ctrls.push_back(new Controller(ctrls_paths[i], nvmNamespace, cudaDevice, queueDepth, numQueues));
            }
        }
    }
};


class SSD_GNN_NVSHMEM_Cache {

    private:
        uint32_t num_ways = 32;
        uint64_t num_sets;
        uint64_t num_pages;
        int num_gpus;
        int local_rank;
        int dim;
        int cache_dim;
        bool track_color = true;
        int64_t* color_buffer_ptr = nullptr;
        int num_color;

        bool is_simulation = false;
        float* sim_buf;
        
    
        NVSHMEM_cache_handle<float>* cache_handle;
        NVSHMEM_cache_d_t<float>* cache_ptr;
        unsigned int* d_request_counters;

        int global_rank;


    public:
        SSD_GNN_NVSHMEM_Cache(SSD_GNN_SSD_Controllers SSD_Controllers, Node_distributor_pybind& node_distributer, int g_rank, int n_gpus, uint64_t cache_size, uint64_t sim_b): 
            num_gpus(n_gpus),
            global_rank(g_rank)
            {
            if(sim_b != 0) 
                is_simulation = true;
            sim_buf = (float*) sim_b;
            color_buffer_ptr = node_distributer.get_color_buffer_ptr();
            num_color = node_distributer.get_num_colors();
            local_rank = SSD_Controllers.cudaDevice;
            dim = SSD_Controllers.dim;
            cache_dim = SSD_Controllers.cache_dim;
            num_pages = cache_size * 1024LL*1024/(SSD_Controllers.page_size);
            num_sets = num_pages / num_ways;
            cache_handle = new NVSHMEM_cache_handle<float>(
                num_sets, num_ways, SSD_Controllers.page_size, SSD_Controllers.ctrls,  global_rank, SSD_Controllers.cudaDevice, num_gpus, track_color, color_buffer_ptr, num_color, is_simulation, sim_buf);
                                                        
                                                        // cpu_ways, is_simulation, sim_buf,
                                                        //     use_color_data, num_colors, color_buffer_ptr);
            cache_ptr = cache_handle -> get_ptr();

            cuda_err_chk(cudaDeviceSynchronize());
            cuda_err_chk(cudaMalloc(&d_request_counters, sizeof(unsigned int) * num_gpus));

            //printf("Init done\n");
        }

        void send_requests(uint64_t i_src_index_ptr, int64_t num_index, uint64_t i_nvshmem_request_ptr, int max_index){
            int64_t* src_index_ptr = (int64_t *)i_src_index_ptr;  
            int64_t* nvshmem_request_ptr = (int64_t *)i_nvshmem_request_ptr;

            uint64_t b_size = 128;
            uint64_t g_size = (num_index+b_size - 1) / b_size;

            cuda_err_chk(cudaMemset((void*)d_request_counters, 0, sizeof(unsigned int) * num_gpus));
            cuda_err_chk(cudaMemset((void*)nvshmem_request_ptr, -1, sizeof(int64_t) * 2 * max_index * num_gpus));
            cuda_err_chk(cudaDeviceSynchronize());
            nvshmem_barrier_all();

            NVSHMEM_send_requests_kernel<<<g_size, b_size>>>(src_index_ptr, num_index, nvshmem_request_ptr + local_rank * max_index * 2, num_gpus, d_request_counters);

            cuda_err_chk(cudaDeviceSynchronize());
            nvshmem_quiet();
            nvshmem_barrier_all();

        }


        void read_feature(uint64_t i_return_tensor_ptr, uint64_t i_nvshmem_index_ptr, int64_t max_index) {
            
            cudaStream_t streams[num_gpus];
            for (int i = 0; i < num_gpus; i++) {
                cudaStreamCreate(&streams[i]);
            }

            float *tensor_ptr = (float *) i_return_tensor_ptr;
            int64_t *nvshmem_index_ptr = (int64_t *)i_nvshmem_index_ptr;


            int b_size = 64;
            int ydim = (num_gpus >= 16) ?  16 : num_gpus;
            dim3 b_dim (b_size, ydim, 1);
            uint64_t g_size = (max_index+b_size - 1) / b_size;
            dim3 g_dim (g_size, 1, 1);

            cuda_err_chk(cudaMemset((void*)d_request_counters, 0, sizeof(unsigned int) * num_gpus));
            NVShmem_count_requests_kernel<<<g_dim, b_dim>>> (nvshmem_index_ptr, d_request_counters, max_index, num_gpus, local_rank);

            cuda_err_chk(cudaDeviceSynchronize());

            unsigned int h_request_counters[num_gpus];
            cuda_err_chk(cudaMemcpy(h_request_counters, d_request_counters, sizeof(unsigned int) * num_gpus,  cudaMemcpyDeviceToHost));
                
         //   printf("dim :%i cache_dim %i\n", (int) dim, (int) cache_dim);
            for (int i = 0; i < num_gpus; i++) {
                uint64_t b_size = 128;
                uint64_t n_warp = b_size / 32;
                uint64_t g_size = (h_request_counters[i]+n_warp - 1) / n_warp;
                NVShmem_read_feature_kernel<<<g_size, b_size, 0, streams[i]>>>(i, cache_ptr, tensor_ptr,
                                                            nvshmem_index_ptr + max_index * 2 * i, dim, h_request_counters[i], cache_dim, local_rank);
            }

            for (int i = 0; i < num_gpus; i++) {
                cudaStreamSynchronize(streams[i]);
            }
            cuda_err_chk(cudaDeviceSynchronize());
            nvshmem_quiet();
            nvshmem_barrier_all();

            return;
        }

        void get_cache_data(int64_t ret_i_ptr){
            int32_t* cache_meta_data = cache_handle->get_color_counter_ptr();
            int32_t* dst = (int32_t*) ret_i_ptr;

            int32_t sum = 0;
            for(int i = 0; i < num_color; i++ ){
                dst[i] = cache_meta_data[i];
            }
            // printf("sum:%i\n", sum);
            return;
        }

        void print_stats(){
            print_stats_kernel<<<1,1>>>(cache_ptr);
        }

        ~SSD_GNN_NVSHMEM_Cache(){
            cudaFree(d_request_counters);
            delete cache_handle;
        }
};


        

class Isolated_Cache {

    private:
        uint32_t num_ways = 32;
        uint64_t num_sets;
        uint64_t num_pages;
        int num_gpus;
        int local_rank;
        int dim;
        int cache_dim;
        bool track_color = true;
        int64_t* color_buffer_ptr = nullptr;
        int num_color;

        bool is_simulation = false;
        float* sim_buf;
        
    
        Isolated_cache_handle<float>* cache_handle;
        Isolated_cache_d_t<float>* cache_ptr;
        unsigned int* d_request_counters;

        int global_rank;


    public:
        Isolated_Cache(SSD_GNN_SSD_Controllers SSD_Controllers, Node_distributor_pybind& node_distributer, int g_rank, int n_gpus, uint64_t cache_size, uint64_t sim_b): 
            num_gpus(n_gpus),
            global_rank(g_rank)
            {
            if(sim_b != 0) 
                is_simulation = true;
            sim_buf = (float*) sim_b;
            color_buffer_ptr = node_distributer.get_color_buffer_ptr();
            num_color = node_distributer.get_num_colors();
            local_rank = SSD_Controllers.cudaDevice;
            dim = SSD_Controllers.dim;
            cache_dim = SSD_Controllers.cache_dim;
            num_pages = cache_size * 1024LL*1024/(SSD_Controllers.page_size);
            num_sets = num_pages / num_ways;
            cache_handle = new Isolated_cache_handle<float>(
                num_sets, num_ways, SSD_Controllers.page_size, SSD_Controllers.ctrls,  global_rank, SSD_Controllers.cudaDevice, num_gpus, track_color, color_buffer_ptr, num_color, is_simulation, sim_buf);
                                                        
                                                        // cpu_ways, is_simulation, sim_buf,
                                                        //     use_color_data, num_colors, color_buffer_ptr);
            cache_ptr = cache_handle -> get_ptr();

            cuda_err_chk(cudaDeviceSynchronize());
            cuda_err_chk(cudaMalloc(&d_request_counters, sizeof(unsigned int) * num_gpus));

            //printf("Init done\n");
        }


        void read_feature(uint64_t i_return_tensor_ptr, uint64_t i_index_ptr, int64_t max_index) {
            
            float *tensor_ptr = (float *) i_return_tensor_ptr;
            int64_t *index_ptr = (int64_t *)i_index_ptr;
        
            int b_size = 64;
            uint64_t g_size = (max_index+b_size - 1) / b_size;
            uint64_t bid = blockIdx.x;

            Isolated_read_feature_kernel<<<g_size, b_size>>>(cache_ptr, tensor_ptr, index_ptr, dim, max_index, cache_dim);
            cuda_err_chk(cudaDeviceSynchronize());
            return;
        }

        void get_cache_data(int64_t ret_i_ptr){
            int32_t* cache_meta_data = cache_handle->get_color_counter_ptr();
            int32_t* dst = (int32_t*) ret_i_ptr;

            int32_t sum = 0;
            for(int i = 0; i < num_color; i++ ){
                dst[i] = cache_meta_data[i];
            }
            // printf("sum:%i\n", sum);
            return;
        }

        void print_stats(){
            print_stats_kernel<<<1,1>>>(cache_ptr);
        }

        ~Isolated_Cache(){
            cudaFree(d_request_counters);
            delete cache_handle;
        }
                          

};



