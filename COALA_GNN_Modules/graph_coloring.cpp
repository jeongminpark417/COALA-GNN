#include "graph_coloring.h"

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

template<bool use_hop_info = false> 
void Graph_Coloring::cpu_flush_buffer(int buffer_id, int hop){
    while (!(bfs_buffers[buffer_id].empty())) {
        auto cur_pair = bfs_buffers[buffer_id].back();
        bfs_buffers[buffer_id].pop_back();
        if(color_buf[cur_pair.first] == 0) {
            color_buf[cur_pair.first] = cur_pair.second;
            if(use_hop_info){
                color_hop_buf[cur_pair.first] =  (uint16_t) (hop+1);
            }
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
    cpu_flush_buffer<true>(next_buffer_id, hop+1);
}

// Optmized code

void Graph_Coloring::cpu_sample_train_nodes(int64_t* train_node_ptr, uint64_t num_train_nodes){

    double train_node_frac = std::min(20.0, (double) num_nodes / (double)num_train_nodes);
    double sampling_rate = global_sampling_rate * train_node_frac;
    printf("sample nodes: %i train nodes:%i sampling rate:%d\n", num_nodes, num_train_nodes, sampling_rate);

    for (uint64_t node = 0; node < num_train_nodes; node++){
        int64_t i = train_node_ptr[node];
        //If the node does not have a color
        if(color_buf[i] == 0) {
            float samp = static_cast<float>(rand()) / RAND_MAX;
            if (samp <= sampling_rate){
                bfs_buffers[0].push_back(std::make_pair(i, color_counter)); // (Node, Color) pair
                color_counter++;
              //  printf("num nodes: %i global sampling rate: %f\n", color_counter, global_sampling_rate);
            }
        }
    }
}

bool Graph_Coloring::is_training_node(uint64_t node_id){
    return training_node_buf[node_id];
}

void Graph_Coloring::update_sample_node_buffer(int64_t* train_node_ptr, uint64_t num_train_nodes){
    
     for (uint64_t node = 0; node < num_train_nodes; node++){
        int64_t i = train_node_ptr[node];
        training_node_buf[i] = true;
     }
}


void Graph_Coloring::cpu_color_graph_optimized(uint64_t i_train_node_ptr, uint64_t num_training_nodes){

    int64_t* train_node_ptr = (int64_t*) i_train_node_ptr;


    update_sample_node_buffer(train_node_ptr, num_training_nodes);
    // First pick sample nodes
    printf("CPU Graph Color Start with color counter:%i \n", color_counter);
    cpu_sample_train_nodes(train_node_ptr, num_training_nodes);
    printf("CPU color graph after sampling color_count:%i global_max hop: %i \n", color_counter, global_max_hop);

    int hop = 0;
    for(; hop < global_max_hop; hop++){
        int cur_buffer_id = hop % 2;
        int next_buffer_id = (hop + 1) % 2;

        //add neighbor nodes if they are 1-hop neighbor of the current trainig nodes
        if(hop == 0){
            size_t initial_size = bfs_buffers[0].size();
            for (size_t i = 0; i < initial_size; i++) {
                auto node_id = bfs_buffers[0][i].first;
                auto node_color = bfs_buffers[0][i].second;
                
                auto start_idx = indptr[node_id];
                auto end_idx = indptr[node_id+1];

                for (auto i = start_idx; i < end_idx; i++){
                    uint64_t neighbor_node = indices[i];
                    if(is_training_node(neighbor_node) && (color_buf[neighbor_node] == 0)){
                        bfs_buffers[0].push_back(std::make_pair(neighbor_node, node_color));
                    }
                }
            }
        }

        //BFS iteration
        while (!(bfs_buffers[cur_buffer_id].empty())) {
            auto cur_pair = bfs_buffers[cur_buffer_id].back();
            bfs_buffers[cur_buffer_id].pop_back();
            //check color
            if(color_buf[cur_pair.first] == 0) {
                //does not have color
                color_buf[cur_pair.first] = cur_pair.second;
                color_hop_buf[cur_pair.first] =  (uint16_t) (hop+1);

                num_colored_nodes++;
                cpu_color_neighbors_csc(next_buffer_id, cur_pair.first, cur_pair.second);
            }
        }
    }
    int next_buffer_id = (hop + 1) % 2;
    cpu_flush_buffer<false>(next_buffer_id, hop+1);
}




template<typename T>
std::vector<std::pair<uint64_t, T>> getTopK(const std::unordered_map<uint64_t, T>& map, size_t K) {
    // Convert the unordered_map to a vector of pairs
    std::vector<std::pair<uint64_t, T>> vec(map.begin(), map.end());

    // Sort the vector based on the values in descending order
    std::sort(vec.begin(), vec.end(), [](const std::pair<uint64_t, T>& a, const std::pair<uint64_t, T>& b) {
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
        auto topk_colors = getTopK<uint64_t>(cur_key_map.second, topk);
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
        auto topk_colors = getTopK<uint64_t>(neigh_map, topk);
        for(int i = 0; i < topk_colors.size(); i++){
            topk_color_buf[(c-1) * topk + i] = topk_colors[i].first;
        }
    }
}



double score_func(int k){
    return std::exp(-0.5 * (double)k);
}

void Graph_Coloring::cpu_calculate_color_affinity(){
    
    std::cout << "Calculating Color affinity\n";
    std::unordered_map<uint64_t,  std::vector<uint64_t>> color_list;

    for (uint64_t node = 0; node < num_nodes; node++){
        auto color = color_buf[node];
        if(color != 0)
            color_list[color].push_back(node);
    }

    auto num_c = color_counter - 1;

    for (uint64_t c = 0; c < num_c; c++){
//        std::vector<uint64_t> neigh_list;
        std::unordered_map<uint64_t, double> neigh_map;
        
        double neigh_count = 0;
        for(const auto& node : color_list[c]){
            auto neighbor_color = color_buf[node];
            auto start_idx = indptr[node];
            auto end_idx = indptr[node+1];

            neigh_count += (end_idx - start_idx);
            for (auto i = start_idx; i < end_idx; i++){
                auto neighbor_color = color_buf[indices[i]];
                auto neighbor_hop = color_hop_buf[indices[i]];
        
                if(neighbor_color != 0 && c != neighbor_color) {
                    neigh_map[neighbor_color] += score_func(neighbor_hop);
                }
            }
        }
        auto topk_colors = getTopK<double>(neigh_map, topk);
        for(int i = 0; i < topk_colors.size(); i++){
            topk_color_buf[(c-1) * topk + i] = topk_colors[i].first;
            if(topk_affinity_buf)
                topk_affinity_buf[(c-1) * topk + i] = (topk_colors[i].second / neigh_count);
        }
    }
}

void Graph_Coloring::set_color_buffer(uint64_t i_ptr){
    color_buf = (uint64_t*) i_ptr;
}

void Graph_Coloring::set_topk_color_buffer(uint64_t i_ptr){
    topk_color_buf = (uint64_t*) i_ptr;
}

void Graph_Coloring::set_topk_affinity_buffer(uint64_t i_ptr){
    topk_affinity_buf = (double*) i_ptr;
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
    training_node_buf = (bool*) malloc(sizeof(bool) * n_nodes);
    if (training_node_buf != nullptr) {
        memset(training_node_buf, 0, sizeof(bool) * n_nodes); // Set all elements to 'false'
    }
    color_hop_buf = (uint16_t*) malloc(sizeof(uint16_t) * n_nodes);

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


    if (training_node_buf != nullptr) {
        free(training_node_buf);  // Free the allocated memory
        training_node_buf = nullptr; // Nullify the pointer for safety
    }
    if(color_hop_buf != nullptr){
        free(color_hop_buf);
        color_hop_buf = nullptr;
    }
    
}




