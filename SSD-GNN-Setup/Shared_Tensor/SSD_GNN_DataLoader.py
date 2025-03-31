from .SSD_GNN_Manager import SSD_GNN_Manager
import torch

#CHECK THIS CLASS
class SSD_GNN_CollateWrapper(object):
    def __init__(self, 
                node_distributor, # Node_Distributor
                refresh_counter = 8 # cache color data refresh time
                ):

    #def __init__(self, NVSHMEM_Cache, Timer, sample_func, g,  device, comm_protocol, reset_counter, num_iterations):
        self.node_distributor = node_distributor

        self.metadata_reuse_counter = 0
        self.refresh_counter = refresh_counter
        self.cache_color_gathered_header = 0

        self.distribute_thread = None
        self.cache_meta_gather_thread = None

    def __call__(self, items):

        # First stage of the Distribution Pipeline
        if(self.distribute_thread == None):
            self.distribute_thread = threading.Thread(target=self.node_distributor.parse_domain_training_nodes, args=(self.cache_color_gathered_header,))
            self.distribute_thread.start()
            self.cur_iteration += 1
                

        self.distribute_thread.join()
        distributed_node_index = self.node_distributor.parsed_training_nodes_buffer[self.node_distributor.parsed_training_nodes_buffer_header]
        self.node_distributor.parsed_training_nodes_buffer_header = (self.node_distributor.parsed_training_nodes_buffer_header + 1) % 2

        if(self.metadata_reuse_counter == self.refresh_counter):
            self.metadata_reuse_counter = 0

            if(self.cache_meta_gather_thread != None):
                self.cache_meta_gather_thread.join()
                self.node_distributor.cache_color_db_header = int((self.node_distributor.cache_color_db_header + 1) % 2)


            #self.NVSHMEM_Cache.get_cache_data(self.cache_meta_tensor.data_ptr())

            self.cache_color_gathered_header = int((self.node_distributor.cache_color_db_header + 1) % 2)

            self.cache_meta_gather_thread = threading.Thread(target=self.node_distributor.gather_cache_meta, args=(self.cache_meta_tensor,))
            self.cache_meta_gather_thread.start()


    #     # distribute Node

   
        if(self.cur_iteration < self.num_iterations):
            self.distribute_thread = threading.Thread(target=self.comm_protocol.parse_domain_training_nodes, args=(self.cache_color_gathered_header,))
            self.distribute_thread.start()
            self.cur_iteration += 1

        graph_device = getattr(self.g, 'device', None)   

        rank_idx =  self.comm_protocol.local_gloo_rank
        idx_list = cur_index[int(rank_idx * self.comm_protocol.batch_size):int((rank_idx+1) * self.comm_protocol.batch_size)]
        idx_list = idx_list.to(self.device)
        idx_list = recursive_apply(idx_list, lambda x: x.to(self.device))
        batch = self.sample_func(self.g, idx_list)

        self.read_counter += 1
        return batch
    
    def clean_up (self):
        if(self.distribute_thread != None):
            self.distribute_thread.join()

        self.cur_iteration = 0

 
        num_ssds,
        num_elems, 
        ssd_read_offset,
        cache_size,  # Cache Size in MB

class SSD_INFO(object):
    def __init__(self, 
                num_ssds,
                num_elems,
                ssd_read_offset):
        self.num_ssds = num_ssds
        self.num_elems = num_elems
        self.ssd_read_offset = ssd_read_offset


class SSD_GNN_DataLoader_Iter(object):
    def __init__(self, dataloader, dataloader_iter):
        self.dataloader = dataloader
        self.dataloader_iter = dataloader_iter
        self.graph_sampler = self.dataloader.graph_sampler

    def __iter__(self):
        return self
    
#    def __next__(self):


class SSD_GNN_DataLoader(torch.utils.data.DataLoader):
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
                 is_simulation = True,
                 shuffle=False,
                 ):

        self.sampler = graph_sampler
        self.batch_size = batch_size

        self.SSD_GNN_Manager = SSD_GNN_Manager(
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
                                    is_simulation=is_simulation
        )
        print("SSD GNN Manager init done")


    def __iter__(self):
        return self
    
    def __del__(self):
        del self.SSD_GNN_Manager

    #def __next__(self):
        

       