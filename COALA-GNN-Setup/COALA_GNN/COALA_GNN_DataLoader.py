from .COALA_GNN_Manager import COALA_GNN_Manager
import torch
import os
import threading
import torch.distributed as dist

#CHECK THIS CLASS
class COALA_GNN_Node_Distribution_Scheduler(object):
    def __init__(self, 
                node_distributor, # Node_Distributor
                ssd_gnn_manager,
                refresh_counter = 8 # cache color data refresh time
                ):

        self.node_distributor = node_distributor
        self.ssd_gnn_manager = ssd_gnn_manager
        self.metadata_reuse_counter = 0
        self.refresh_counter = refresh_counter
        self.cache_color_gathered_header = 0

        self.distribute_thread = None
        self.cache_meta_gather_thread = None
         
        self.cache_meta_tensor = torch.zeros(self.node_distributor.num_colors, dtype=torch.int32)


    def run(self, is_last : bool):

        if(self.node_distributor.comm_manager.is_master):
            # First stage of the Distribution Pipeline
            if(self.distribute_thread == None):
                self.distribute_thread = threading.Thread(target=self.node_distributor.parse_domain_training_nodes, args=(self.cache_color_gathered_header,))
                self.distribute_thread.start()

                # self.node_distributor.parse_domain_training_nodes(self.cache_color_gathered_header)
                # self.distribute_thread =  1
                    
            self.distribute_thread.join()

        distributed_node_index = self.node_distributor.parsed_training_nodes_buffer[self.node_distributor.parsed_training_nodes_buffer_header]
        master_process_id = self.node_distributor.comm_manager.master_process_id

        self.node_distributor.comm_manager.broadcast_training_nodes(distributed_node_index)
        self.node_distributor.parsed_training_nodes_buffer_header = (self.node_distributor.parsed_training_nodes_buffer_header + 1) % 2

        if(self.metadata_reuse_counter == self.refresh_counter):
            self.metadata_reuse_counter = 0

            if(self.cache_meta_gather_thread != None):
                self.cache_meta_gather_thread.join()
                self.node_distributor.cache_color_db_header = int((self.node_distributor.cache_color_db_header + 1) % 2)


            self.ssd_gnn_manager.COALA_GNN_Cache.get_cache_data(self.cache_meta_tensor.data_ptr())

            self.cache_color_gathered_header = int((self.node_distributor.cache_color_db_header + 1) % 2)

            self.cache_meta_gather_thread = threading.Thread(target=self.node_distributor.gather_cache_meta, args=(self.cache_meta_tensor,))
            self.cache_meta_gather_thread.start()

            # self.cache_meta_gather_thread = 1
            # self.node_distributor.gather_cache_meta(self.cache_meta_tensor)

        if(self.node_distributor.comm_manager.is_master):
            if(is_last == False):
                self.distribute_thread = threading.Thread(target=self.node_distributor.parse_domain_training_nodes, args=(self.cache_color_gathered_header,))
                self.distribute_thread.start()
                #self.node_distributor.parse_domain_training_nodes(self.cache_color_gathered_header)


        self.metadata_reuse_counter += 1
        local_r = self.node_distributor.comm_manager.local_rank 
        distributed_node_index_per_GPU = distributed_node_index[(local_r * self.node_distributor.batch_size):((local_r+1) * self.node_distributor.batch_size)]
       # print(f"Rank: {self.node_distributor.comm_manager.local_rank} distributed node idx: {distributed_node_index_per_GPU}")
        return distributed_node_index_per_GPU


 

class SSD_INFO(object):
    def __init__(self, 
                num_ssds,
                page_size,
                num_elems,
                ssd_read_offset):
        self.num_ssds = num_ssds
        self.num_elems = num_elems
        self.ssd_read_offset = ssd_read_offset
        self.page_size = page_size


class COALA_GNN_DataLoader(torch.utils.data.DataLoader):
    def __init__(self,
                 SSD_info, #SSD_INFO CLASS
                 node_distributor, # Node_Distributor
                 graph,
                 graph_sampler, 
                 batch_size,
                 dim,
                 fan_out,
                 cache_size, # In MB
                 device,
                 cache_backend = "nvshmem",
                 sim_buf = None,
                 shuffle=False,
                 ):

        self.sampler = graph_sampler
        self.batch_size = batch_size
        self.g = graph
        self.SSD_info = SSD_info
        self.cache_backend = cache_backend
        self.node_distributor = node_distributor
        self.device = device

        self.COALA_GNN_Manager = COALA_GNN_Manager(
                                    node_distributor = node_distributor,
                                    page_size = SSD_info.page_size,
                                    num_ssds = SSD_info.num_ssds,
                                    num_elems = SSD_info.num_elems,
                                    ssd_read_offset = SSD_info.ssd_read_offset,
                                    cache_size = cache_size,
                                    batch_size = batch_size,
                                    fan_out = fan_out,
                                    dim = dim,
                                    MPI_comm_manager = node_distributor.comm_manager,
                                    device = device,
                                    cache_backend = cache_backend,
                                    sim_buf=sim_buf
        )

        self.scheduler = COALA_GNN_Node_Distribution_Scheduler(
                            node_distributor = self.node_distributor,
                            ssd_gnn_manager = self.COALA_GNN_Manager,
                            refresh_counter = 8
                        )
        self.counter = 0
        self.index_len = len(self.node_distributor.index_tensor)
        self.total_count = int(self.index_len / self.node_distributor.global_batch_size) - 1
        print(f"Index len: {self.index_len}")
        
    def __iter__(self):
        return self

    # Return Tuple
    # (Input Nodes, seeds, blocks, feature data)
    def __next__(self):
        #if (self.counter >= self.index_len):
        if (self.counter == self.total_count):
            self.node_distributor.reset()
            self.counter = 0
            raise StopIteration  # This tells Python to stop iteration
        is_last_iter = False
        if(self.counter + self.node_distributor.global_batch_size >= self.index_len):
            is_last_iter = True
        

        #index = self.node_distributor.index_tensor[self.counter:(self.counter+self.batch_size)]
        distributed_index = self.scheduler.run(is_last_iter).to(self.device)
        batch = self.sampler.sample(self.g, distributed_index)

        #self.counter += self.node_distributor.global_batch_size
        self.counter += 1

        return self.COALA_GNN_Manager.fetch_feature(batch)
    

    def print_stats(self):
        self.COALA_GNN_Manager.print_stats()

    def __del__(self):
        del self.COALA_GNN_Manager
        
