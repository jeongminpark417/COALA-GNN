#ifndef GRAPH_COLORING_H
#define GRAPH_COLORING_H

//#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>
#include <vector>
#include <cstdint> 
#include <cstdlib> 

#include <iostream>

struct Graph_Coloring {
    uint64_t num_nodes = 0;
    uint64_t num_colored_nodes = 0;
    uint64_t color_counter = 1;
    int step_num = 1;
    int global_max_hop = 10;
    float global_sampling_rate = 0.005;

    int topk = 10;
    void cpu_color_graph();
    void cpu_color_graph_optimized(uint64_t i_train_node_ptr, uint64_t num_training_nodes);
    void cpu_sample_nodes();
    void cpu_sample_train_nodes(int64_t* train_node_ptr, uint64_t num_train_nodes);
   
    void cpu_color_neighbors_csc(int, uint64_t, uint64_t);
    template<bool use_hop_info> 
    void cpu_flush_buffer(int, int);
    void cpu_count_nearest_color();
    void cpu_count_nearest_color_less_memory();
    void cpu_calculate_color_affinity();

    void set_color_buffer(uint64_t);
    void set_topk_color_buffer(uint64_t);
    void set_topk_affinity_buffer(uint64_t);
    
    void set_adj_csc(uint64_t, uint64_t);

    uint64_t get_num_color();
    uint64_t get_num_color_node();

    bool is_training_node(uint64_t);
    void update_sample_node_buffer(int64_t* train_node_ptr, uint64_t num_train_nodes);




    uint64_t* color_buf = nullptr;
    uint64_t* topk_color_buf = nullptr; //(color x topk)
    double* topk_affinity_buf = nullptr; //(color x topk)

    uint16_t* color_hop_buf = nullptr;
    bool* training_node_buf = nullptr;

    //CSC Graph ADJ Matrix
    uint64_t* indptr = nullptr;
    uint64_t* indices = nullptr;

    Graph_Coloring(uint64_t);
    ~Graph_Coloring();
    private:
        std::vector<std::pair<uint64_t, uint64_t>> bfs_buffers[2];
        std::unordered_map<uint64_t, std::unordered_map<uint64_t, uint64_t>> color_connectivity_map;

};

#endif
