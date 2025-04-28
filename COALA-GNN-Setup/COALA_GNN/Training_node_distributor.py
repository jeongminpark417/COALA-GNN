from COALA_GNN_Pybind import Node_distributor_pybind
import torch
import torch.distributed as dist

class Node_Distributor(object):
    def __init__(self, 
                 comm_manager, #MPI_Comm_Manager
                 index_tensor,
                 batch_size,
                 color_file: str,
                 topk_file: str,
                 score_file: str,
                 parsing_method = "node_color",
                ):
        self.index_tensor = index_tensor.to("cpu").contiguous()
        self.index_offset = 0
        self.parsing_method = parsing_method
        self.batch_size = batch_size
        self.comm_manager = comm_manager

        self.domain_batch_size = batch_size * comm_manager.local_size
        self.global_batch_size = batch_size * comm_manager.global_size

        self.distribute_manager = Node_distributor_pybind(self.index_tensor.data_ptr(), self.comm_manager.node_id, self.batch_size, comm_manager.local_size, comm_manager.num_master_process, color_file, topk_file, score_file)
        self.num_colors = self.distribute_manager.get_num_colors()

        self.parsed_training_nodes_buffer = []
        self.parsed_training_nodes_buffer.append(torch.zeros(self.domain_batch_size, dtype=torch.int64).contiguous())
        self.parsed_training_nodes_buffer.append(torch.zeros(self.domain_batch_size, dtype=torch.int64).contiguous())
        self.parsed_training_nodes_buffer_header = 0

        self.cache_color_double_buffer = []
        self.cache_color_double_buffer.append([torch.zeros(self.num_colors, dtype=torch.int32) for _ in range(comm_manager.num_master_process)])
        self.cache_color_double_buffer.append([torch.zeros(self.num_colors, dtype=torch.int32) for _ in range(comm_manager.num_master_process)])
        self.cache_color_db_header = 0

    def gather_cache_meta(self, gpu_cache_meta):
        self.comm_manager.gather_cache_meta(gpu_cache_meta, self.cache_color_double_buffer[self.cache_color_db_header])

    def parse_domain_training_nodes(self, color_buf_read_header):
        if(self.parsing_method == "baseline"):
            node_id = self.comm_manager.master_process_index
            self.parsed_training_nodes_buffer[self.parsed_training_nodes_buffer_header] = self.index_tensor[(self.index_offset + node_id*self.domain_batch_size):(self.index_offset + (node_id + 1) * self.domain_batch_size)]
            self.index_offset += self.global_batch_size
            #print(f"parsed node: {self.parsed_training_nodes_buffer[self.parsed_training_nodes_buffer_header]}")

            return self.parsed_training_nodes_buffer[self.parsed_training_nodes_buffer_header]

        elif(self.parsing_method == "node_color"):
            gather_ptr = []
            for gt in self.cache_color_double_buffer[color_buf_read_header]:
                gather_ptr.append(gt.data_ptr())
            
            self.distribute_manager.distribute_node_with_affinity(self.index_offset, self.parsed_training_nodes_buffer[self.parsed_training_nodes_buffer_header].data_ptr(), gather_ptr)

            self.index_offset += self.global_batch_size
#            print(f"parsed node: {self.parsed_training_nodes_buffer[self.parsed_training_nodes_buffer_header]}")
            return self.parsed_training_nodes_buffer[self.parsed_training_nodes_buffer_header]
        else:
            print(f"Unsupported parsing method: {self.parsing_method}")

    def reset(self):
        self.index_offset = 0
        self.parsed_training_nodes_buffer_header = 0
        self.cache_color_db_header = 0
