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
#include "shared_UVA.cuh"
#include "nvshmem_manager.cuh"

#define BLOCK_SIZE 256
namespace py = pybind11;


PYBIND11_MODULE(SSD_GNN_Pybind, m) {
    py::class_<SharedUVAManager>(m, "SharedUVAManager")
        .def(py::init<const std::string&, int64_t, int, int64_t, int64_t>())
        .def("get_host_ptr", &SharedUVAManager::get_host_ptr)
        .def("get_device_ptr", &SharedUVAManager::get_device_ptr)
        .def("cleanup", &SharedUVAManager::cleanup);

    py::class_<SSD_GNN_SSD_Controllers>(m, "SSD_GNN_SSD_Controllers")
        .def(py::init<uint32_t, uint64_t, uint64_t, uint32_t, int, bool>());

    py::class_<SSD_GNN_NVSHMEM_Cache>(m, "SSD_GNN_NVSHMEM_Cache")
        .def(py::init<SSD_GNN_SSD_Controllers, int, uint64_t, uint64_t>())
        .def("send_requests", &SSD_GNN_NVSHMEM_Cache::send_requests)
        .def("read_feature", &SSD_GNN_NVSHMEM_Cache::read_feature);

    py::class_<Node_distributor_pybind>(m, "Node_distributor_pybind")
        .def(py::init<uint64_t, int>()) 
        .def(py::init<uint64_t, int, int, int, int, const std::string&, const std::string&, const std::string&>())  
        .def("distribute_node_with_affinity", &Node_distributor_pybind::distribute_node_with_affinity)
        .def("get_num_colors", &Node_distributor_pybind::get_num_colors);

    py::class_<NVSHMEM_Manager>(m, "NVSHMEM_Manager")
        .def(py::init<int64_t, int>()) 
        .def("allocate", &NVSHMEM_Manager::allocate)
        .def("free", &NVSHMEM_Manager::free)
        .def("finalize", &NVSHMEM_Manager::finalize);



}
