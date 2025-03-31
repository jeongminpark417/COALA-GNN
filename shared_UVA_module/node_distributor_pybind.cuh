#pragma once

#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <stdio.h>
#include <regex>

char* load_file_to_memory(const std::string& file_path, size_t& file_size) {
    std::ifstream file(file_path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Unable to open file: " + file_path);
    }

    // Get the size of the file
    file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Allocate buffer memory using malloc
    char* buffer;
    cuda_err_chk(cudaHostAlloc((void**)&buffer, file_size, cudaHostAllocMapped));
    if (buffer == nullptr) {
        throw std::runtime_error("Failed to allocate memory for file buffer.");
    }

    // Read file into the buffer
    if (!file.read(buffer, file_size)) {
        free(buffer);  // Free the buffer if reading fails
        throw std::runtime_error("Error reading the file: " + file_path);
    }

    return buffer;
}

template<typename T>
void  parse_numpy_file(const std::string &color_file, int dim, char*& ret_buffer, T*& ret_buffer_ptr, std::vector<int>& shape){
  size_t file_size;
  
  char* buffer = load_file_to_memory(color_file, file_size);
  char* buffer_ptr = buffer;
  std::string magic_string(buffer_ptr, 6);
    if (magic_string != "\x93NUMPY") {
        throw std::runtime_error("Not a valid .npy file.");
    }

  uint8_t major_version = buffer_ptr[6];
  uint8_t minor_version = buffer_ptr[7];
  std::cout << "Version: " << static_cast<int>(major_version) << "." << static_cast<int>(minor_version) << std::endl;
  buffer_ptr += 8; // Move past the magic string and version bytes

  // Step 3: Read the header length based on the version
  uint32_t header_len = 0;
  if (major_version == 1) {
      // Version 1.x uses 2 bytes for header length
      uint16_t header_len_16;
      memcpy(&header_len_16, buffer_ptr, sizeof(uint16_t));
      header_len = header_len_16;
      buffer_ptr += sizeof(uint16_t);  // Move past the header length
  } else if (major_version == 2) {
      // Version 2.x uses 4 bytes for header length
      memcpy(&header_len, buffer_ptr, sizeof(uint32_t));
      buffer_ptr += sizeof(uint32_t);  // Move past the header length
  } else {
      throw std::runtime_error("Unsupported .npy file version: " + std::to_string(major_version));
  }


  std::string header(buffer_ptr, header_len);
  buffer_ptr += header_len;  // Move the pointer past the header

  // Step 5: Parse the header using regex to extract shape, dtype, and fortran order
  std::regex shape_regex_1D(R"('shape':\s?\((\d+),?\))");
  std::regex shape_regex_2D(R"('shape':\s?\((\d+),\s?(\d+)\))");
  std::regex dtype_regex(R"('descr':\s*'(.*?)')");
  std::regex fortran_regex(R"('fortran_order':\s?(True|False))");

  std::smatch match;

  // Extract shape
  if(dim == 1){
    if (std::regex_search(header, match, shape_regex_1D)) {
        int shape_value = std::stoi(match[1].str());
        shape.push_back(shape_value);
    }
  }
  else if(dim == 2){
    if (std::regex_search(header, match, shape_regex_2D)) {
        int shape_value1 = std::stoi(match[1].str());
        int shape_value2 = std::stoi(match[2].str());
        shape.push_back(shape_value1);
        shape.push_back(shape_value2);
    }
  }
  else{
    throw std::runtime_error("Unsupported dimension for .npy file\n");
  }

  if (std::regex_search(header, match, dtype_regex)) {
      auto dtype= match[1].str();
      std::cout << "dtype: " << dtype << std::endl;
      assert((dtype == "<i8" || dtype == "<f8") && "The dtype is not '<i8'. This program only supports 64-bit integers.");
  }

  ret_buffer = buffer;
  ret_buffer_ptr = (T*) buffer_ptr;

  return;
}

// index dtype should be int64_t
class Node_distributor_pybind {
    private:
    int64_t* item_ptr;
    int64_t offset = 0;
    int node_id, num_nodes, local_size;
    int batch_size, domain_batch_size, global_batch_size;

    char* color_ptr = nullptr;
    int64_t* color_buffer_ptr = nullptr;
    int num_colors = 0;

    char* topk_ptr = nullptr;
    int64_t* topk_buffer_ptr = nullptr;
    std::vector<int> topk_shape, color_shape, score_shape;
  
    char* score_ptr = nullptr;
    double* score_buffer_ptr = nullptr;

    bool use_color_information = false;

    public:
    Node_distributor_pybind(uint64_t i_item_ptr, int n_nodes) : num_nodes(n_nodes){
        item_ptr = (int64_t*) i_item_ptr;
    }

    // Adding color information
    Node_distributor_pybind(uint64_t i_item_ptr, int n_id, int b_size, int local_size_, int n_nodes, const std::string &color_file, const std::string &topk_file, const std::string &score_file) : node_id(n_id), batch_size(b_size), local_size(local_size_), num_nodes(n_nodes){
        item_ptr = (int64_t*) i_item_ptr;
        use_color_information = true;
        parse_numpy_file<int64_t>(color_file, 1, color_ptr, color_buffer_ptr, color_shape);
        parse_numpy_file<int64_t>(topk_file, 2, topk_ptr, topk_buffer_ptr, topk_shape);
        num_colors = topk_shape[0];
        parse_numpy_file<double>(score_file, 1, score_ptr, score_buffer_ptr, score_shape);

        domain_batch_size = batch_size * local_size;
        global_batch_size = domain_batch_size * num_nodes;
    }

    void distribute_node_with_affinity(uint64_t i_parsed_ptr, const std::vector<uint64_t>&  meta_data_list){

        if(score_ptr == nullptr || topk_ptr == nullptr || color_ptr == nullptr){
            std::cout << "Error: Node_distributor_pybind is not created with color information.\n";
            return;
        }
        int item_len = global_batch_size;

        int bucket_len[num_nodes] =  {0};  
        int64_t* return_ptr = (int64_t*) i_parsed_ptr;
        int topk = topk_shape[1];

        int cur_max_part = 0;
        double max_score = -1.0;

        for(int i = 0; i < item_len; i++){
            int64_t node_id = item_ptr[i];

            int64_t node_color = color_buffer_ptr[node_id];
            cur_max_part = 0;
            max_score = -1.0;

        for(int j = 0; j < num_nodes; j++){
            int32_t* meta_ptr = (int32_t*)(meta_data_list[j]);
            double cur_score = 0;
            double prev_score = 0;
            double prev_neigh_score = 0;
            double prev_affinity = 0;

            if(node_color == 0){
                cur_score = 0.0;
            } 
            else{
                for(int k = 0; k < topk; k++){
                    int64_t neigh_color = topk_buffer_ptr[(node_color - 1) * topk + k];
                    double neigh_affinity = score_buffer_ptr[(node_color - 1) * topk + k];
                    double neigh_score = 0.0;
                    if(neigh_color != 0){
                    if(meta_ptr[neigh_color] == 0 )
                        continue;
                    neigh_score =  (double) (meta_ptr[neigh_color]);
                    prev_score = cur_score;
                    cur_score += (neigh_score * neigh_affinity );
                    }
                    if(neigh_score < 0.0 || neigh_affinity < 0.0){
                    int32_t neigh_score_int = (meta_ptr[neigh_color]);
                    }
                    prev_affinity = neigh_affinity;
                    prev_neigh_score = neigh_score;
                }
            }
        
            double before_score = cur_score;

            if(bucket_len[j] == batch_size)
            cur_score = -1.0;

            if(cur_score > max_score){
            cur_max_part = j;
            max_score = cur_score;
            }
      }
      

     if(cur_max_part == node_id){
       return_ptr[bucket_len[cur_max_part]] = node_id;
     }
      bucket_len[cur_max_part]+=1;
    }
  }

  int get_num_colors(){
    return num_colors;
  }
};

