#ifndef GRAPH_COLORING_H
#define GRAPH_COLORING_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>
#include <vector>
#include <cstdint> 
#include <cstdlib> 



struct Graph_Coloring {
    uint64_t num_nodes = 0;
    uint64_t num_colored_nodes = 0;
    uint64_t color_counter = 0;
    int step_num = 1;
    int global_max_hop = 1;
    float global_sampling_rate = 0.001;

    int topk = 10;
    void cpu_color_graph();
    void cpu_sample_nodes();
    void cpu_color_neighbors_csc(int, uint64_t, uint64_t);
    void cpu_flush_buffer(int);
    void cpu_count_nearest_color();

    void set_color_buffer(uint64_t);
    void set_topk_color_buffer(uint64_t);
    uint64_t get_num_color();

    uint64_t* color_buf = nullptr;
    uint64_t* topk_color_buf = nullptr; //(color x topk)

    //CSC Graph ADJ Matrix
    uint64_t* indptr = nullptr;
    uint64_t* indices = nullptr;

    private:
        std::vector<std::pair<uint64_t, uint64_t>> bfs_buffers[2];
        std::unordered_map<uint64_t, std::unordered_map<uint64_t, uint64_t>> color_connectivity_map;

};

#endif