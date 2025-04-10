#pragma once

#include <mpi4py/mpi4py.h>
#include <mpi.h>
#include "nvshmem.h"
#include "nvshmemx.h"


class NVSHMEM_Manager {

    // private:
    // nvshmemx_init_attr_t attr;

    public:
    NVSHMEM_Manager(int64_t local_comm_ptr, int local_rank){
        auto l_comm_ptr = (MPI_Comm*)(local_comm_ptr);

        cuda_err_chk(cudaSetDevice(local_rank));
        
        //MPI_Comm mpi_comm = MPI_COMM_WORLD;
        
        nvshmemx_init_attr_t attr;
        attr.mpi_comm = l_comm_ptr;
        //attr.mpi_comm = &mpi_comm;

        //nvshmem_init();
        printf("nvshmem init start\n");
        nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
        printf("nvshmem init done\n");

    }

    uint64_t allocate(int size){
        void* destination = (void *) nvshmem_malloc(size);
        if (destination == nullptr) {
            fprintf(stderr, "PE %d: nvshmem_malloc failed!\n", nvshmem_my_pe());
            return 0;
        }
        uint64_t int_ptr = (uint64_t) destination;
        return int_ptr;
    }
    
    void free(uint64_t dest_ptr) {
        void* destination = (void*) dest_ptr;
        if (destination) {
            nvshmem_free(destination);

            destination = nullptr;
        }
    }
 
    void finalize(){
        nvshmem_finalize();
    }
};




