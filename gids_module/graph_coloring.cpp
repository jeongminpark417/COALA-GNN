
#include <graph_coloring.h>

void Graph_Coloring::cpu_sample_nodes(){
    for (uint64_t i = 0; i < num_nodes; i++){
        //If the node does not have a color
        if(color_buf[i] == UINT64_MAX) {
            float samp = static_cast<float>(rand()) / RAND_MAX;
            if (samp <= global_sampling_rate){
                bfs_buffers[0].push_back(std::make_pair(i, color_counter)); // (Node, Color) pair
                color_counter++;
            }
        }
    }
}

void Graph_Coloring::cpu_color_neighbors_csc(int buffer_id, uint64_t node, uint64_t color){
    auto start_idx = indptr[node];
    auto end_idx = indptr[node+1];

    for (auto i = start_idx; i < end_idx; i++){
        bfs_buffers[buffer_id].push_back(std::make_pair(indices[i], color));
    }

}

void Graph_Coloring::cpu_flush_buffer(int buffer_id){
    while (!(bfs_buffers[buffer_id].empty())) {
        auto cur_pair = bfs_buffers[buffer_id].back();
        bfs_buffers[buffer_id].pop_back();
        if(color_buf[cur_pair.first] == UINT64_MAX) {
            color_buf[cur_pair.first] = cur_pair.second;
            num_colored_nodes++;
        }
    }
}

void Graph_Coloring::cpu_color_graph(){

    // First pick sample nodes
    cpu_sample_nodes();
    int hop = 0;
    for(; hop < global_max_hop; hop++){
        int cur_buffer_id = hop % 2;
        int next_buffer_id = (hop + 1) % 2;

        //BFS iteration
        while (!(bfs_buffers[cur_buffer_id].empty())) {
            auto cur_pair = bfs_buffers[cur_buffer_id].back();
            bfs_buffers[cur_buffer_id].pop_back();
            //check color
            if(color_buf[cur_pair.first] == UINT64_MAX) {
                //does not have color
                color_buf[cur_pair.first] = cur_pair.second;
                num_colored_nodes++;
                cpu_color_neighbors_csc(next_buffer_id, cur_pair.first, cur_pair.second);
            }
            
        }
    }
    int next_buffer_id = (hop + 1) % 2;
    cpu_flush_buffer(next_buffer_id);
}



std::vector<std::pair<uint64_t, uint64_t>> getTopK(const std::unordered_map<uint64_t, uint64_t>& map, size_t K) {
    // Convert the unordered_map to a vector of pairs
    std::vector<std::pair<uint64_t, uint64_t>> vec(map.begin(), map.end());

    // Sort the vector based on the values in descending order
    std::sort(vec.begin(), vec.end(), [](const std::pair<uint64_t, uint64_t>& a, const std::pair<uint64_t, uint64_t>& b) {
        return a.second > b.second;
    });

    // Get the top K elements
    if (vec.size() > K) {
        vec.resize(K);
    }

    return vec;
}

void Graph_Coloring::cpu_count_nearest_color(){
    for (uint64_t node = 0; node < num_nodes; node++){

            auto color = color_buf[node];

            //CSC Version
            auto start_idx = indptr[node];
            auto end_idx = indptr[node+1];

            for (auto i = start_idx; i < end_idx; i++){
                auto neighbor_color = color_buf[indices[i]];
                if(color != neighbor_color) {
                    color_connectivity_map[color][neighbor_color] += 1;
                }
            }
    }

        for (const auto& cur_key_map : color_connectivity_map) {
            auto cur_color = cur_key_map.first;
            auto topk_colors = getTopK(cur_key_map.second, topk);
            //color count
            for(int i = 0; i < topk; i++){
                topk_color_buf[cur_color * topk + i] = topk_colors[i].second;
            }
        }      
}


void Graph_Coloring::set_color_buffer(uint64_t i_ptr){
    color_buf = (uint64_t*) i_ptr;
}

void Graph_Coloring::set_topk_color_buffer(uint64_t i_ptr){
    topk_color_buf = (uint64_t*) i_ptr;
}


void Graph_Coloring::set_adj_csc(uint64_t i_indp, uint64_t i_indices){
    indptr =  (uint64_t*) i_indp;
    indices = (uint64_t*) i_indices;
}


uint64_t Graph_Coloring::get_num_color(){
    return color_counter;
}

PYBIND11_MODULE(Graph_Coloring, m) {
  m.doc() = "Graph Coloring Libraray";

  namespace py = pybind11;

    py::class_<Graph_Coloring>(m, "Graph_Coloring")
    .def(py::init<>())
    .def("cpu_color_graph", &Graph_Coloring::cpu_color_graph)
    .def("cpu_count_nearest_color", &Graph_Coloring::cpu_count_nearest_color)
    .def("set_color_buffer", &Graph_Coloring::set_color_buffer)
    .def("set_topk_color_buffer", &Graph_Coloring::set_topk_color_buffer)
    .def("set_adj_csc", &Graph_Coloring::set_adj_csc)
    .def("get_num_color", &Graph_Coloring::get_num_color);
}

