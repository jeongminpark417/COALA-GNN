
#ifndef __NVSHMEMCACHE_CACHE_BUFFER_H__
#define __NVSHMEMCACHE_CACHE_BUFFER_H__

#include "buffer.h"

inline DmaPtr createDmaHost(const nvm_ctrl_t* ctrl, size_t size)
{
    nvm_dma_t* dma = nullptr;

    void* buffer = nullptr;
    void* origPtr = nullptr;
    
    
    size_t padded_size = size +  64*1024;
    cudaError_t err = cudaHostAlloc(&origPtr, padded_size, cudaHostAllocDefault);
    if (err != cudaSuccess)
    {
        throw error(string("Failed to allocate host memory: ") + cudaGetErrorString(err));
    }
    buffer = (void*) ((((uint64_t)origPtr) + (64*1024))  & 0xffffffffff0000);


   // int err  = posix_memalign(&buffer, 4096, size);

    if (err) {
        throw error(string("Failed to allocate host memory: ") + std::to_string(err));
    }
    int status = nvm_dma_map_host(&dma, ctrl, buffer, size);
    //int status = nvm_dma_map_host(&dma, ctrl, origPtr, size);

    if (!nvm_ok(status))
    {
        //cudaFreeHost(buffer);
        free(buffer);
        throw error(string("Failed to map host memory: ") + nvm_strerror(status));
    }

    return DmaPtr(dma, [buffer, origPtr](nvm_dma_t* dma) {
        nvm_dma_unmap(dma);
        cudaFreeHost(origPtr);
        //free(buffer);

    });
}


#endif
