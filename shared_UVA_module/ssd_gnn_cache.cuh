#pragma once
#include <stdexcept>  // for std::runtime_error
#include <ctrl.h>  // for NVME controller
#include "nvshmem_cache.h"


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


    SSD_GNN_SSD_Controllers(uint32_t num_ctrls, uint64_t n_elems, uint64_t read_off, uint32_t device_id, int feat_dim, bool sim)
        : n_ctrls(num_ctrls), num_elements(n_elems), cudaDevice(device_id), offset(read_off), dim(feat_dim), SSD_SIM(sim){
        
        if(dim <= 512)
            cache_dim = 512;
        else if(dim <= 1024)
            cache_dim = 1024;
        else if(dim <= 2048)
            cache_dim = 2048;
        else if (dim <= 4096)
            cache_dim = 4096;
        else if (dim <= 8192)
            cache_dim = 8192;
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

    public:
        SSD_GNN_NVSHMEM_Cache(SSD_GNN_SSD_Controllers SSD_Controllers, int n_gpus, uint64_t cache_size): num_gpus(n_gpus){
      
        num_pages = cache_size * 1024LL*1024/(SSD_Controllers.page_size);
        num_sets = num_pages / num_ways;
        auto cache_handle = new NVSHMEM_cache_handle<float, true>(
            num_sets, num_ways, SSD_Controllers.page_size, SSD_Controllers.ctrls, SSD_Controllers.cudaDevice, num_gpus);
                                                    
                                                    // cpu_ways, is_simulation, sim_buf,
                                                    //     use_color_data, num_colors, color_buffer_ptr);
        auto cache_ptr = cache_handle -> get_ptr();

        cuda_err_chk(cudaDeviceSynchronize());
        // cuda_err_chk(cudaMalloc(&d_request_counters, sizeof(unsigned int) * n_gpus));

        printf("Init done\n");
   
    }

};



