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
#include <mpi.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#define BLOCK_SIZE 256

//MPI_INIT should have be called before creating this obj
class SharedUVAManager {
    private:
        std::string shm_full_path;
        void* mmap_ptr;
        void* dev_ptr;
        int shm_fd;
        int shm_id;
        MPI_Comm comm;
        int global_rank;
        int local_rank;
        int node_id;
        int64_t SHM_SIZE;

    public:
        //MPI_COMM might need to be fixed
        SharedUVAManager(const std::string& path, int64_t shm_size, int node) 
            : shm_full_path(path), SHM_SIZE(shm_size), mmap_ptr(nullptr), dev_ptr(nullptr), shm_fd(-1), node_id(node){

            MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
            MPI_Comm_split(MPI_COMM_WORLD, node_id, global_rank, &comm);
            MPI_Comm_rank(comm, &local_rank);
            initialize_shared_memory();
        }

        ~SharedUVAManager() {
            cleanup();
        }

        void initialize_shared_memory() {
            if(local_rank == 0){
                shm_fd = shm_open(shm_full_path.c_str(), O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
                if(shm_fd < 0){
                    std::cerr << "shm_open failed, Reason=" << strerror(errno) << std::endl;
                }
                exit(1);
                ftruncate(shm_fd, SHM_SIZE);
                MPI_Barrier(comm);
            }
            else{
                MPI_Barrier(comm);
                shm_fd = shm_open(shm_full_path.c_str(), O_RDWR, S_IRUSR | S_IWUSR);
            }

            mmap_ptr = mmap(nullptr, SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
            if(mmap_ptr == MAP_FAILED){
                std::cerr << "mmap failed, Reason= " << strerror(errno) << std::endl;
                exit(1);
            }

            cudaError_t err = cudaHostRegister(mmap_ptr, SHM_SIZE, cudaHostRegisterDefault);
            if(err != cudaSuccess){
                std::cerr << "cudaHostRegister ..." << std::endl;
                exit(1);
            }

            err = cudaHostGetDevicePointer(&dev_ptr, mmap_ptr, 0);
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
    void* get_host_ptr() { return mmap_ptr; }
    void* get_device_ptr() { return dev_ptr; }
};



namespace py = pybind11;

PYBIND11_MODULE(shared_memory, m) {
    py::class_<SharedUVAManager>(m, "SharedUVAManager")
        .def(py::init<const std::string&, int64_t, int>())
        .def("get_host_ptr", &SharedUVAManager::get_host_ptr)
        .def("get_device_ptr", &SharedUVAManager::get_device_ptr)
        .def("cleanup", &SharedUVAManager::cleanup);
}
