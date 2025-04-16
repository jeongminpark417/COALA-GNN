#pragma once

#include <cuda_runtime.h>
#include <fcntl.h>      
#include <sys/stat.h>   
#include <sys/mman.h>
#include <errno.h>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <mpi4py/mpi4py.h>
#include <mpi.h>

#include "ssd_gnn_cache.cuh"
#include "node_distributor_pybind.cuh"


#define BLOCK_SIZE 256
namespace py = pybind11;

//MPI_INIT should have be called before creating this obj
class SharedUVAManager {
    private:
        std::string shm_full_path;
        void* mmap_ptr;
        void* dev_ptr;
        int shm_fd;
        int shm_id;
        MPI_Comm global_comm;
        MPI_Comm local_comm;
        int global_rank;
        int local_rank;
        int node_id;
        int64_t SHM_SIZE;

    public:
        //py_comm is MPI_COMM*
        SharedUVAManager(const std::string& path, int64_t shm_size, int node, int64_t global_comm_ptr, int64_t local_comm_ptr) 
            : shm_full_path(path), SHM_SIZE(shm_size), mmap_ptr(nullptr), dev_ptr(nullptr), shm_fd(-1), node_id(node){

            auto g_comm_ptr = (MPI_Comm*)(global_comm_ptr);
            auto l_comm_ptr = (MPI_Comm*)(local_comm_ptr);

            global_comm = *g_comm_ptr;
            local_comm = *l_comm_ptr;

            int initialized;
            MPI_Initialized(&initialized);
            if (!initialized) {
                throw std::runtime_error("MPI must be initialized before calling this function.");
            }
            MPI_Comm_rank(global_comm, &global_rank);
            MPI_Comm_rank(local_comm, &local_rank);
            initialize_shared_memory();
        }

        ~SharedUVAManager() {
            cleanup();
        }

        void initialize_shared_memory() {
            printf("shared memory setting device %i\n", (int) local_rank);
            cudaSetDevice(local_rank);

            if(local_rank == 0){
                shm_fd = shm_open(shm_full_path.c_str(), O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
                if(shm_fd < 0){
                    std::cerr << "shm_open failed, Reason=" << strerror(errno) << std::endl;
                    exit(1);
                }
                ftruncate(shm_fd, SHM_SIZE);
                MPI_Barrier(local_comm);
            }
            else{
                MPI_Barrier(local_comm);
                shm_fd = shm_open(shm_full_path.c_str(), O_RDWR, S_IRUSR | S_IWUSR);
            }

            mmap_ptr = mmap(nullptr, SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
            if(mmap_ptr == MAP_FAILED){
                std::cerr << "mmap failed, Reason= " << strerror(errno) << std::endl;
                exit(1);
            }

            cudaError_t err = cudaHostRegister(mmap_ptr, SHM_SIZE, cudaHostRegisterDefault);
            if(err != cudaSuccess){
                std::cerr << "cudaHostRegister error" << std::endl;
                exit(1);
            }
            err = cudaHostGetDevicePointer(&dev_ptr, mmap_ptr, 0);
            if(err != cudaSuccess){
                std::cerr << "cudaHostGetDevicePointer error" << std::endl;
                exit(1);
            }
            memset(mmap_ptr, 0, SHM_SIZE);
        }

    void cleanup() {
        if (mmap_ptr) {
            cudaHostUnregister(mmap_ptr);
            munmap(mmap_ptr, SHM_SIZE);
        }
        if (local_rank == 0) {
            shm_unlink(shm_full_path.c_str());
        }
    }
    uintptr_t get_host_ptr() { return reinterpret_cast<uintptr_t>(mmap_ptr); }
    uintptr_t get_device_ptr() { return reinterpret_cast<uintptr_t>(dev_ptr); }


};


