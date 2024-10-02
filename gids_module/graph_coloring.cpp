
#include <graph_coloring.h>

void Graph_Coloring::cpu_sample_nodes(){
    printf("sample nodes: %i\n", num_nodes);
    for (uint64_t i = 0; i < num_nodes; i++){
        //If the node does not have a color

        if(color_buf[i] == 0) {
            float samp = static_cast<float>(rand()) / RAND_MAX;
            if (samp <= global_sampling_rate){
                bfs_buffers[0].push_back(std::make_pair(i, color_counter)); // (Node, Color) pair
                color_counter++;
              //  printf("num nodes: %i global sampling rate: %f\n", color_counter, global_sampling_rate);
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
        if(color_buf[cur_pair.first] == 0) {
            color_buf[cur_pair.first] = cur_pair.second;
            num_colored_nodes++;
        }
    }
}

void Graph_Coloring::cpu_color_graph(){

    // First pick sample nodes
    printf("CPU color graph color_count:%i \n", color_counter);
    cpu_sample_nodes();
    printf("CPU color graph after sampling color_count:%i global_max hop: %i \n", color_counter, global_max_hop);

    int hop = 0;
    for(; hop < global_max_hop; hop++){
        int cur_buffer_id = hop % 2;
        int next_buffer_id = (hop + 1) % 2;

        //BFS iteration
        while (!(bfs_buffers[cur_buffer_id].empty())) {
            auto cur_pair = bfs_buffers[cur_buffer_id].back();
            bfs_buffers[cur_buffer_id].pop_back();
            //check color
            if(color_buf[cur_pair.first] == 0) {
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
    
            if(neighbor_color != 0 && color != neighbor_color) {
                color_connectivity_map[color][neighbor_color] += 1;
            }
        }
    }
    for (const auto& cur_key_map : color_connectivity_map) {
        auto cur_color = cur_key_map.first;
        auto topk_colors = getTopK(cur_key_map.second, topk);
        //color count
        for(int i = 0; i < topk_colors.size(); i++){
            topk_color_buf[(cur_color-1) * topk + i] = topk_colors[i].first;
        }
    } 
}


void Graph_Coloring::cpu_count_nearest_color_less_memory(){
    
    std::unordered_map<uint64_t,  std::vector<uint64_t>> color_list;

    for (uint64_t node = 0; node < num_nodes; node++){
        auto color = color_buf[node];
        if(color != 0)
            color_list[color].push_back(node);
    }

    auto num_c = color_counter - 1;

    for (uint64_t c = 0; c < num_c; c++){
//        std::vector<uint64_t> neigh_list;
        std::unordered_map<uint64_t, uint64_t> neigh_map;

        for(const auto& node : color_list[c]){
            auto neighbor_color = color_buf[node];
            auto start_idx = indptr[node];
            auto end_idx = indptr[node+1];

            for (auto i = start_idx; i < end_idx; i++){
                auto neighbor_color = color_buf[indices[i]];
        
                if(neighbor_color != 0 && c != neighbor_color) {
                    neigh_map[neighbor_color] += 1;
                }
            }
        }
        auto topk_colors = getTopK(neigh_map, topk);
        for(int i = 0; i < topk_colors.size(); i++){
            topk_color_buf[(c-1) * topk + i] = topk_colors[i].first;
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
    if(color_counter == 0) return 0;
    return color_counter - 1;
}

uint64_t Graph_Coloring::get_num_color_node(){

	return num_colored_nodes;
}


Graph_Coloring::Graph_Coloring(uint64_t n_nodes){
	num_nodes = n_nodes;
}

Graph_Coloring::~Graph_Coloring() {
    printf("Destructing Graph Coloring Strucutre\n");
    for (int i = 0; i < 2; ++i) {
        bfs_buffers[i].clear();
        std::vector<std::pair<uint64_t, uint64_t>>().swap(bfs_buffers[i]);
    }

    // Clear and release memory for `color_connectivity_map`
    color_connectivity_map.clear();
    std::unordered_map<uint64_t, std::unordered_map<uint64_t, uint64_t>>().swap(color_connectivity_map);

}


PYBIND11_MODULE(Graph_Coloring, m) {
  m.doc() = "Graph Coloring Libraray";

  namespace py = pybind11;

    py::class_<Graph_Coloring>(m, "Graph_Coloring")
    .def(py::init<uint64_t>())
    .def("cpu_color_graph", &Graph_Coloring::cpu_color_graph)
    .def("cpu_count_nearest_color", &Graph_Coloring::cpu_count_nearest_color)
    .def("cpu_count_nearest_color_less_memory", &Graph_Coloring::cpu_count_nearest_color_less_memory)
    .def("set_color_buffer", &Graph_Coloring::set_color_buffer)
    .def("set_topk_color_buffer", &Graph_Coloring::set_topk_color_buffer)
    .def("set_adj_csc", &Graph_Coloring::set_adj_csc)
    .def("get_num_color_node", &Graph_Coloring::get_num_color_node)
    .def("get_num_color", &Graph_Coloring::get_num_color);
}

