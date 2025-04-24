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
#include "graph_coloring.cpp"

#define BLOCK_SIZE 256

namespace py = pybind11;


PYBIND11_MODULE(COALA_GNN_Pybind, m) {
    py::class_<SharedUVAManager>(m, "SharedUVAManager")
        .def(py::init<const std::string&, int64_t, int, int64_t, int64_t>())
        .def("get_host_ptr", &SharedUVAManager::get_host_ptr)
        .def("get_device_ptr", &SharedUVAManager::get_device_ptr)
        .def("cleanup", &SharedUVAManager::cleanup);

    py::class_<SSD_GNN_SSD_Controllers>(m, "SSD_GNN_SSD_Controllers")
        .def(py::init<uint32_t, uint32_t, uint64_t, uint64_t, uint32_t, int, bool>());

    py::class_<SSD_GNN_NVSHMEM_Cache>(m, "SSD_GNN_NVSHMEM_Cache")
        .def(py::init<SSD_GNN_SSD_Controllers, Node_distributor_pybind&, int, int, uint64_t, uint64_t>())
        .def("send_requests", &SSD_GNN_NVSHMEM_Cache::send_requests)
        .def("read_feature", &SSD_GNN_NVSHMEM_Cache::read_feature)
        .def("get_cache_data", &SSD_GNN_NVSHMEM_Cache::get_cache_data)
        .def("print_stats", &SSD_GNN_NVSHMEM_Cache::print_stats);

    py::class_<Isolated_Cache>(m, "Isolated_Cache")
        .def(py::init<SSD_GNN_SSD_Controllers, Node_distributor_pybind&, int, int, uint64_t, uint64_t>())
        .def("read_feature", &Isolated_Cache::read_feature)
        .def("get_cache_data", &Isolated_Cache::get_cache_data)
        .def("print_stats", &Isolated_Cache::print_stats);
        
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

    py::class_<Graph_Coloring>(m, "Graph_Coloring")
        .def(py::init<uint64_t>())
        .def("cpu_color_graph", &Graph_Coloring::cpu_color_graph)
        .def("cpu_color_graph_optimized", &Graph_Coloring::cpu_color_graph_optimized)
        .def("cpu_count_nearest_color", &Graph_Coloring::cpu_count_nearest_color)
        .def("cpu_count_nearest_color_less_memory", &Graph_Coloring::cpu_count_nearest_color_less_memory)
        .def("cpu_calculate_color_affinity", &Graph_Coloring::cpu_calculate_color_affinity)
        .def("set_color_buffer", &Graph_Coloring::set_color_buffer)
        .def("set_topk_color_buffer", &Graph_Coloring::set_topk_color_buffer)
        .def("set_topk_affinity_buffer", &Graph_Coloring::set_topk_affinity_buffer)
        .def("set_adj_csc", &Graph_Coloring::set_adj_csc)
        .def("get_num_color_node", &Graph_Coloring::get_num_color_node)
        .def("get_num_color", &Graph_Coloring::get_num_color);

}
