import argparse, time
import numpy as np
import torch
import os.path as osp

import dgl
from dgl.data import DGLDataset
import warnings
warnings.filterwarnings("ignore")

from Shared_Tensor import MPI_Comm_Manager, Shared_UVA_Tensor_Manager
import cupy as cp
from mpi4py import MPI
import gc



class IGB260M(object):
    def __init__(self, root: str, size: str, in_memory: int, uva_graph: int,  \
            classes: int, synthetic: int, emb_size: int, data: str):
        self.dir = root
        self.size = size
        self.synthetic = synthetic
        self.in_memory = in_memory
        self.num_classes = classes
        self.emb_size = emb_size
        self.uva_graph = uva_graph
        self.data = data
    
    def num_nodes(self):
        if self.data == 'OGB':
            return 111059956

        if self.size == 'tiny':
            return 100000
        elif self.size == 'small':
            return 1000000
        elif self.size == 'medium':
            return 10000000
        elif self.size == 'large':
            return 100000000
        elif self.size == 'full':
            return 269346174

    @property
    def paper_feat(self) -> np.ndarray:
        num_nodes = self.num_nodes()
        # TODO: temp for bafs. large and full special case
        if self.data == 'OGB':
            path = osp.join(self.dir, 'node_feat.npy')
            if self.in_memory:
                emb = np.load(path)
            else:
                emb = np.load(path, mmap_mode='r')

        elif self.size == 'large' or self.size == 'full':
        #     path = '/mnt/nvme17/node_feat.npy'
        #     #path = '/mnt/raid0_2/node_feat.npy'
            path = osp.join(self.dir, self.size, 'processed', 'paper', 'node_feat.npy')
            if self.in_memory:
                emb = np.memmap(path, dtype='float32', mode='r',  shape=(num_nodes,1024)).copy()
            else:    
                emb = np.memmap(path, dtype='float32', mode='r',  shape=(num_nodes,1024))
        else:
            path = osp.join(self.dir, self.size, 'processed', 'paper', 'node_feat.npy')
            if self.synthetic:
                emb = np.random.rand(num_nodes, 1024).astype('f')
            else:
                if self.in_memory:
                    emb = np.load(path)
                else:
                    emb = np.load(path, mmap_mode='r')

        return emb

    @property
    def paper_label(self) -> np.ndarray:

        if(self.data == 'OGB'):
            return np.random.randint(low=0, size=111059956, high=171)
        elif self.size == 'large' or self.size == 'full':
            num_nodes = self.num_nodes()
            if self.num_classes == 19:
            #    path = '/mnt/nvme16/IGB260M_part_2/full/processed/paper/node_label_19_extended.npy'
                path = osp.join(self.dir, self.size, 'processed', 'paper', 'node_label_19.npy')
                if(self.in_memory):
                    node_labels = np.memmap(path, dtype='float32', mode='r',  shape=(num_nodes)).copy()
                else:
                    node_labels = np.memmap(path, dtype='float32', mode='r',  shape=(num_nodes))
                # Actual number 227130858
            else:
                #path = '/mnt/nvme16/IGB260M_part_2/full/processed/paper/node_label_2K_extended.npy'
                path = osp.join(self.dir, self.size, 'processed', 'paper', 'node_label_2K.npy')
                if(self.in_memory):
                    node_labels = np.load(path)
                else:
                    node_labels = np.memmap(path, dtype='float32', mode='r',  shape=(num_nodes))
                # Actual number 157675969

        else:
            if self.num_classes == 19:
                path = osp.join(self.dir, self.size, 'processed', 'paper', 'node_label_19.npy')
            else:
                path = osp.join(self.dir, self.size, 'processed', 'paper', 'node_label_2K.npy')
            if self.in_memory:
                node_labels = np.load(path)
            else:
                node_labels = np.load(path, mmap_mode='r')
        return node_labels

    @property
    def paper_edge(self) -> np.ndarray:
        path = osp.join(self.dir, self.size, 'processed', 'paper__cites__paper', 'edge_index.npy')
        if self.data == 'OGB':
            path = osp.join(self.dir, 'edge_index.npy')
        # elif self.size == 'full':
        #     path = '/mnt/nvme16/IGB260M_part_2/full/processed/paper__cites__paper/edge_index.npy'
        # elif self.size == 'large':
        #     path = '/mnt/nvme7/large/processed/paper__cites__paper/edge_index.npy'
        
        if self.in_memory or self.uva_graph:
            return np.load(path)
        else:
            return np.load(path, mmap_mode='r')


    


class IGB260MDGLDataset(DGLDataset):
    def __init__(self, args):
        self.dir = args.path
        self.args = args
        super().__init__(name='IGB260MDGLDataset')

    def process(self):
        dataset = IGB260M(root=self.dir, size=self.args.dataset_size, in_memory=self.args.in_memory, uva_graph=self.args.uva_graph, \
            classes=self.args.num_classes, synthetic=self.args.synthetic, emb_size=self.args.emb_size, data=self.args.data)

        node_features = torch.from_numpy(dataset.paper_feat)
        node_edges = torch.from_numpy(dataset.paper_edge)
        node_labels = torch.from_numpy(dataset.paper_label).to(torch.long)

        print("node edge:", node_edges)
        #cur_path = osp.join(self.dir, self.args.dataset_size, 'processed')
        # cur_path = '/mnt/nvme16/IGB260M_part_2/full/processed'
        # edge_row_idx = torch.from_numpy(np.load(cur_path + '/paper__cites__paper/edge_index_csc_row_idx.npy'))
        # edge_col_idx = torch.from_numpy(np.load(cur_path + '/paper__cites__paper/edge_index_csc_col_idx.npy'))
        # edge_idx = torch.from_numpy(np.load(cur_path + '/paper__cites__paper/edge_index_csc_edge_idx.npy'))
        
        # if self.args.dataset_size == 'full':   
        #     self.graph = dgl.graph(('csc', (edge_col_idx,edge_row_idx,edge_idx)), num_nodes=node_features.shape[0])
        #     self.graph  = self.graph.formats('csc') 
        # else:
        
        self.graph = dgl.graph((node_edges[:, 0],node_edges[:, 1]), num_nodes=node_features.shape[0])
        print("self graph: ", self.graph.formats()) 
        print("skipping feat")
        self.graph.ndata['feat'] = node_features
        

        self.graph.ndata['label'] = node_labels
        print("self graph2: ", self.graph.formats())
        if self.args.dataset_size != 'full':
            self.graph = dgl.remove_self_loop(self.graph)
            self.graph = dgl.add_self_loop(self.graph)
        print("self graph3: ", self.graph.formats())
        
        if self.args.dataset_size == 'full':
            #TODO: Put this is a meta.pt file
            if self.args.num_classes == 19:
                n_labeled_idx = 227130858
            else:
                n_labeled_idx = 157675969

            n_nodes = node_features.shape[0]
            n_train = int(n_labeled_idx * 0.6)
            n_val   = int(n_labeled_idx * 0.2)
            print("self graph4: ", self.graph.formats())    
            train_mask = torch.zeros(n_nodes, dtype=torch.bool)
            val_mask = torch.zeros(n_nodes, dtype=torch.bool)
            test_mask = torch.zeros(n_nodes, dtype=torch.bool)
            
            train_mask[:n_train] = True
            val_mask[n_train:n_train + n_val] = True
            test_mask[n_train + n_val:n_labeled_idx] = True
            print("self graph5: ", self.graph.formats())
            self.graph.ndata['train_mask'] = train_mask
            self.graph.ndata['val_mask'] = val_mask
            self.graph.ndata['test_mask'] = test_mask
        else:
            n_nodes = node_features.shape[0]
            n_train = int(n_nodes * 0.6)
            n_val   = int(n_nodes * 0.2)
            
            train_mask = torch.zeros(n_nodes, dtype=torch.bool)
            val_mask = torch.zeros(n_nodes, dtype=torch.bool)
            test_mask = torch.zeros(n_nodes, dtype=torch.bool)
            
            train_mask[:n_train] = True
            val_mask[n_train:n_train + n_val] = True
            test_mask[n_train + n_val:] = True
            
            self.graph.ndata['train_mask'] = train_mask
            self.graph.ndata['val_mask'] = val_mask
            self.graph.ndata['test_mask'] = test_mask
        
    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return len(self.graphs)

class IGBDatast_Shared_UVA(DGLDataset):
    def __init__(self, args, comm_manager, device):
        self.dir = args.path
        self.size = args.dataset_size
        self.num_classes = args.num_classes
        self.args = args
        self.comm_manager = comm_manager
        self.device = device
        self.feat_data = None
        self.local_rank = comm_manager.local_rank
        super().__init__(name='IGB260M')

    def num_nodes(self):
        if self.size == 'tiny':
            return 100000
        elif self.size == 'small':
            return 1000000
        elif self.size == 'medium':
            return 10000000
        elif self.size == 'large':
            return 100000000
        elif self.size == 'full':
            return 269346174

    def get_label(self) -> np.ndarray:
        if self.size == 'large' or self.size == 'full':
            num_nodes = self.num_nodes()
            if self.num_classes == 19:
            #    path = '/mnt/nvme16/IGB260M_part_2/full/processed/paper/node_label_19_extended.npy'
                path = osp.join(self.dir, self.size, 'processed', 'paper', 'node_label_19.npy')
                node_labels = np.memmap(path, dtype='float32', mode='r',  shape=(num_nodes))
                # Actual number 227130858
            else:
                #path = '/mnt/nvme16/IGB260M_part_2/full/processed/paper/node_label_2K_extended.npy'
                path = osp.join(self.dir, self.size, 'processed', 'paper', 'node_label_2K.npy')
                node_labels = np.load(path)
     
        else:
            if self.num_classes == 19:
                path = osp.join(self.dir, self.size, 'processed', 'paper', 'node_label_19.npy')
            else:
                path = osp.join(self.dir, self.size, 'processed', 'paper', 'node_label_2K.npy')
            node_labels = np.load(path)
        return node_labels

    def process(self):
        if(self.args.feat_cpu):

            feat_array_size = np.empty([1], dtype="int")
            feat_shape_list = np.empty([2], dtype="int")
            emb = None
            if(self.local_rank == 0):
                print("Loading feature data into CPU for SSD simulation")
                feat_path = osp.join(self.dir, self.size, 'processed', 'paper', 'node_feat.npy')
                emb = np.load(feat_path)
                feat_array_size[0] = emb.nbytes
                feat_shape_list = np.array(emb.shape)

            #Sharing feat information
            self.comm_manager.local_comm.Barrier()
            self.comm_manager.local_comm.Bcast(feat_array_size, root = 0)

            self.comm_manager.local_comm.Barrier()
            self.comm_manager.local_comm.Bcast(feat_shape_list, root = 0)
            feat_shape = torch.Size(feat_shape_list)


            self.shared_UVA_manager_feat = Shared_UVA_Tensor_Manager(self.comm_manager, "/shared_mem_feat", feat_array_size[0])
            self.feat_data = self.shared_UVA_manager_feat.get_tensor(dtype=cp.float32, tensor_shape=feat_shape)
            self.shared_UVA_manager_feat.write_np_array(self.feat_data, emb)
            #self.shared_UVA_manager_feat.write_np_array_gpu(self.feat_data, emb, self.device)

            del emb

        edge_array_size = np.empty([1], dtype="int")
        edge_shape_list = np.empty([2], dtype="int")
        edge_array = None
        if(self.local_rank == 0):
            edge_path = osp.join(self.dir, self.size, 'processed', 'paper__cites__paper', 'edge_index.npy')
            edge_array = np.load(edge_path)
            print(f"edge type: {edge_array.dtype} Bytes: {edge_array.nbytes}, Shape: {edge_array.shape}")
            edge_array_size[0] = edge_array.nbytes
            edge_shape_list = np.array(edge_array.shape)

        #Sharing edge information
        self.comm_manager.local_comm.Barrier()
        self.comm_manager.local_comm.Bcast(edge_array_size, root = 0)

        self.comm_manager.local_comm.Barrier()
        self.comm_manager.local_comm.Bcast(edge_shape_list, root = 0)
        edge_shape = torch.Size(edge_shape_list)


        self.shared_UVA_manager = Shared_UVA_Tensor_Manager(self.comm_manager, "/shared_mem", edge_array_size[0])
        node_edges = self.shared_UVA_manager.get_tensor(dtype=cp.int64, tensor_shape=edge_shape)
        print(f"node edges shape: {node_edges.shape} device: {node_edges.device} array: {node_edges}" )
        self.shared_UVA_manager.write_np_array(node_edges, edge_array)
        print(f"After COPY node edges shape: {node_edges.shape} device: {node_edges.device} array: {node_edges}" )
        del edge_array

        gc.collect()
        print("Freed local graph data")

        node_labels = torch.from_numpy(self.get_label()).to(torch.long).to(self.device)

        n_nodes = self.num_nodes()
        print("Number of Nodes: ", n_nodes)
        self.graph = dgl.graph((node_edges[:, 0],node_edges[:, 1]), num_nodes=n_nodes)
        self.graph.ndata['label'] = node_labels
        self.graph  = self.graph.formats('csc')
        print(self.graph.formats())


        if self.args.dataset_size == 'full':
            #TODO: Put this is a meta.pt file
            if self.args.num_classes == 19:
                n_labeled_idx = 227130858
            else:
                n_labeled_idx = 157675969

            n_train = int(n_labeled_idx * 0.6)
            n_val   = int(n_labeled_idx * 0.2)
            train_mask = torch.zeros(n_nodes, dtype=torch.bool).to(self.device)
            val_mask = torch.zeros(n_nodes, dtype=torch.bool).to(self.device)
            test_mask = torch.zeros(n_nodes, dtype=torch.bool).to(self.device)
            
            train_mask[:n_train] = True
            val_mask[n_train:n_train + n_val] = True
            test_mask[n_train + n_val:n_labeled_idx] = True
            self.graph.ndata['train_mask'] = train_mask
            self.graph.ndata['val_mask'] = val_mask
            self.graph.ndata['test_mask'] = test_mask
        else:
            n_train = int(n_nodes * 0.6)
            n_val   = int(n_nodes * 0.2)
            
            train_mask = torch.zeros(n_nodes, dtype=torch.bool).to(self.device)
            val_mask = torch.zeros(n_nodes, dtype=torch.bool).to(self.device)
            test_mask = torch.zeros(n_nodes, dtype=torch.bool).to(self.device)
            
            train_mask[:n_train] = True
            val_mask[n_train:n_train + n_val] = True
            test_mask[n_train + n_val:] = True
            
            self.graph.ndata['train_mask'] = train_mask
            self.graph.ndata['val_mask'] = val_mask
            self.graph.ndata['test_mask'] = test_mask
        
    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return len(self.graphs)


class IGBDatast_Shared_CSC_UVA(DGLDataset):


    def num_nodes(self):
        if self.size == 'tiny':
            return 100000
        elif self.size == 'small':
            return 1000000
        elif self.size == 'medium':
            return 10000000
        elif self.size == 'large':
            return 100000000
        elif self.size == 'full':
            return 269346174

    def get_label(self) -> np.ndarray:
        if self.size == 'large' or self.size == 'full':
            num_nodes = self.num_nodes()
            if self.num_classes == 19:
            #    path = '/mnt/nvme16/IGB260M_part_2/full/processed/paper/node_label_19_extended.npy'
                path = osp.join(self.dir, self.size, 'processed', 'paper', 'node_label_19.npy')
                node_labels = np.memmap(path, dtype='float32', mode='r',  shape=(num_nodes))
                # Actual number 227130858
            else:
                #path = '/mnt/nvme16/IGB260M_part_2/full/processed/paper/node_label_2K_extended.npy'
                path = osp.join(self.dir, self.size, 'processed', 'paper', 'node_label_2K.npy')
                node_labels = np.load(path)
     
        else:
            if self.num_classes == 19:
                path = osp.join(self.dir, self.size, 'processed', 'paper', 'node_label_19.npy')
            else:
                path = osp.join(self.dir, self.size, 'processed', 'paper', 'node_label_2K.npy')
            node_labels = np.load(path)
        return node_labels
    
    def __init__(self, args, comm_manager, device):
        self.dir = args.path
        self.size = args.dataset_size
        self.num_classes = args.num_classes
        self.args = args
        self.comm_manager = comm_manager
        self.device = device
        self.feat_data = None
        self.local_rank = comm_manager.local_rank
    #     super().__init__(name='IGB260M')

    # def process(self):
        if(self.args.feat_cpu):

            feat_array_size = np.empty([1], dtype="int")
            feat_shape_list = np.empty([2], dtype="int")
            emb = None
            if(self.local_rank == 0):
                print("Loading feature data into CPU for SSD simulation")
                feat_path = osp.join(self.dir, self.size, 'processed', 'paper', 'node_feat.npy')
                emb = np.load(feat_path)
                feat_array_size[0] = emb.nbytes
                feat_shape_list = np.array(emb.shape)

            #Sharing feat information
            self.comm_manager.local_comm.Barrier()
            self.comm_manager.local_comm.Bcast(feat_array_size, root = 0)

            self.comm_manager.local_comm.Barrier()
            self.comm_manager.local_comm.Bcast(feat_shape_list, root = 0)
            feat_shape = torch.Size(feat_shape_list)


            self.shared_UVA_manager_feat = Shared_UVA_Tensor_Manager(self.comm_manager, "/shared_mem_feat", feat_array_size[0])
            self.feat_data = self.shared_UVA_manager_feat.get_tensor(dtype=np.float32, tensor_shape=feat_shape, device=self.device)
            self.shared_UVA_manager_feat.write_np_array(self.feat_data, emb)
            #self.shared_UVA_manager_feat.write_np_array_gpu(self.feat_data, emb, self.device)

            del emb

        csc_indptr_array_size = np.empty([1], dtype="int")
        csc_indptr_shape_list = np.empty([1], dtype="int")
        csc_indptr_array = None

        csc_indices_array_size = np.empty([1], dtype="int")
        csc_indices_shape_list = np.empty([1], dtype="int")
        csc_indices_array = None

        csc_edge_ids_array_size = np.empty([1], dtype="int")
        csc_edge_ids_shape_list = np.empty([1], dtype="int")
        csc_edge_ids_array = None

        if(self.local_rank == 0):
            #(indptr, indices, edge_ids)
            indptr_path = osp.join(self.dir, self.size, 'processed', 'paper__cites__paper', 'csc_indptr.npy')
            csc_indptr_array = np.load(indptr_path)
            print(f"edge type: {csc_indptr_array.dtype} Bytes: {csc_indptr_array.nbytes}, Shape: {csc_indptr_array.shape}")
            csc_indptr_array_size[0] = csc_indptr_array.nbytes
            csc_indptr_shape_list = np.array(csc_indptr_array.shape)

            indices_path = osp.join(self.dir, self.size, 'processed', 'paper__cites__paper', 'csc_indices.npy')
            csc_indices_array = np.load(indices_path)
            print(f"edge type: {csc_indices_array.dtype} Bytes: {csc_indices_array.nbytes}, Shape: {csc_indices_array.shape}")
            csc_indices_array_size[0] = csc_indices_array.nbytes
            csc_indices_shape_list = np.array(csc_indices_array.shape)

            edge_ids_path = osp.join(self.dir, self.size, 'processed', 'paper__cites__paper', 'csc_edge_ids.npy')
            csc_edge_ids_array = np.load(edge_ids_path)
            print(f"edge type: {csc_edge_ids_array.dtype} Bytes: {csc_edge_ids_array.nbytes}, Shape: {csc_edge_ids_array.shape}")
            csc_edge_ids_array_size[0] = csc_edge_ids_array.nbytes
            csc_edge_ids_shape_list = np.array(csc_edge_ids_array.shape)

        #Sharing edge information
        self.comm_manager.local_comm.Barrier()
        self.comm_manager.local_comm.Bcast(csc_indptr_array_size, root = 0)

        self.comm_manager.local_comm.Barrier()
        self.comm_manager.local_comm.Bcast(csc_indptr_shape_list, root = 0)
        csc_indptr_shape = torch.Size(csc_indptr_shape_list)

        self.comm_manager.local_comm.Barrier()
        self.comm_manager.local_comm.Bcast(csc_indices_array_size, root = 0)

        self.comm_manager.local_comm.Barrier()
        self.comm_manager.local_comm.Bcast(csc_indices_shape_list, root = 0)
        csc_indices_shape = torch.Size(csc_indices_shape_list)

        self.comm_manager.local_comm.Barrier()
        self.comm_manager.local_comm.Bcast(csc_edge_ids_array_size, root = 0)

        self.comm_manager.local_comm.Barrier()
        self.comm_manager.local_comm.Bcast(csc_edge_ids_shape_list, root = 0)
        csc_edge_ids_shape = torch.Size(csc_edge_ids_shape_list)


        self.shared_UVA_manager_indptr = Shared_UVA_Tensor_Manager(self.comm_manager, "/shared_mem_1", csc_indptr_array_size[0])
        csc_indptr = self.shared_UVA_manager_indptr.get_tensor(dtype=np.int64, tensor_shape=csc_indptr_shape, device=self.device)
        print(f"node edges shape: {csc_indptr.shape} device: {csc_indptr.device} array: {csc_indptr}" )
        self.shared_UVA_manager_indptr.write_np_array(csc_indptr, csc_indptr_array)
        print(f"After COPY node edges shape: {csc_indptr.shape} device: {csc_indptr.device} array: {csc_indptr}" )
        del csc_indptr_array

        self.shared_UVA_manager_indicies = Shared_UVA_Tensor_Manager(self.comm_manager, "/shared_mem_2", csc_indices_array_size[0])
        csc_indicies = self.shared_UVA_manager_indicies.get_tensor(dtype=np.int64, tensor_shape=csc_indices_shape, device=self.device)
        print(f"node edges shape: {csc_indicies.shape} device: {csc_indicies.device} array: {csc_indicies}" )
        self.shared_UVA_manager_indicies.write_np_array(csc_indicies, csc_indices_array)
        print(f"After COPY node edges shape: {csc_indicies.shape} device: {csc_indicies.device} array: {csc_indicies}" )
        del csc_indices_array

        self.shared_UVA_manager_edge_id = Shared_UVA_Tensor_Manager(self.comm_manager, "/shared_mem_3", csc_edge_ids_array_size[0])
        edge_ids_indicies = self.shared_UVA_manager_edge_id.get_tensor(dtype=np.int64, tensor_shape=csc_edge_ids_shape, device=self.device)
        print(f"node edges shape: {edge_ids_indicies.shape} device: {edge_ids_indicies.device} array: {edge_ids_indicies}" )
        self.shared_UVA_manager_edge_id.write_np_array(edge_ids_indicies, csc_edge_ids_array)
        print(f"After COPY node edges shape: {edge_ids_indicies.shape} device: {edge_ids_indicies.device} array: {edge_ids_indicies}" )
        del csc_edge_ids_array

        gc.collect()
        print("Freed local graph data")

        node_labels = torch.from_numpy(self.get_label()).to(torch.long).to(self.device)

        n_nodes = self.num_nodes()
        self.graph = dgl.graph(('csc',(csc_indptr, csc_indicies, edge_ids_indicies)), num_nodes=n_nodes, idtype=torch.int64, device=self.device)
        self.graph  = self.graph.formats('csc')
        self.graph.ndata['label'] = node_labels
        print(self.graph.formats())



        if self.args.dataset_size == 'full':
            #TODO: Put this is a meta.pt file
            if self.args.num_classes == 19:
                n_labeled_idx = 227130858
            else:
                n_labeled_idx = 157675969

            n_train = int(n_labeled_idx * 0.6)
            n_val   = int(n_labeled_idx * 0.2)
            train_mask = torch.zeros(n_nodes, dtype=torch.bool).to(self.device)
            val_mask = torch.zeros(n_nodes, dtype=torch.bool).to(self.device)
            test_mask = torch.zeros(n_nodes, dtype=torch.bool).to(self.device)
            
            train_mask[:n_train] = True
            val_mask[n_train:n_train + n_val] = True
            test_mask[n_train + n_val:n_labeled_idx] = True
            self.graph.ndata['train_mask'] = train_mask
            self.graph.ndata['val_mask'] = val_mask
            self.graph.ndata['test_mask'] = test_mask
        else:
            n_train = int(n_nodes * 0.6)
            n_val   = int(n_nodes * 0.2)
            
            train_mask = torch.zeros(n_nodes, dtype=torch.bool).to(self.device)
            val_mask = torch.zeros(n_nodes, dtype=torch.bool).to(self.device)
            test_mask = torch.zeros(n_nodes, dtype=torch.bool).to(self.device)
            
            train_mask[:n_train] = True
            val_mask[n_train:n_train + n_val] = True
            test_mask[n_train + n_val:] = True
            
            self.graph.ndata['train_mask'] = train_mask
            self.graph.ndata['val_mask'] = val_mask
            self.graph.ndata['test_mask'] = test_mask
        
    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return len(self.graphs)


class OGBDataset_Shared_UVA(DGLDataset):
    def __init__(self, args, comm_manager, device):
        self.dir = args.path
        self.args = args
        self.comm_manager = comm_manager
        self.device = device
        self.feat_data = None
        self.local_rank = comm_manager.local_rank


   # def process(self):
        if(self.args.feat_cpu):

            feat_array_size = np.empty([1], dtype="int")
            feat_shape_list = np.empty([2], dtype="int")
            emb = None
            if(self.local_rank == 0):
                print("Loading feature data into CPU for SSD simulation")
                feat_path = osp.join(self.dir, 'raw', 'node_feat.npy')
                emb = np.load(feat_path)
                feat_array_size[0] = emb.nbytes
                feat_shape_list = np.array(emb.shape)

            #Sharing feat information
            self.comm_manager.local_comm.Barrier()
            self.comm_manager.local_comm.Bcast(feat_array_size, root = 0)

            self.comm_manager.local_comm.Barrier()
            self.comm_manager.local_comm.Bcast(feat_shape_list, root = 0)
            feat_shape = torch.Size(feat_shape_list)


            self.shared_UVA_manager_feat = Shared_UVA_Tensor_Manager(self.comm_manager, "/shared_mem_feat", feat_array_size[0])
            self.feat_data = self.shared_UVA_manager_feat.get_tensor(dtype=cp.float32, tensor_shape=feat_shape)
            self.shared_UVA_manager_feat.write_np_array(self.feat_data, emb)
            del emb


        edge_array_size = np.empty([1], dtype="int")
        edge_shape_list = np.empty([2], dtype="int")
        edge_array = None
        if(self.local_rank == 0):
            edge_path = osp.join(self.dir, 'raw' ,'edge_index.npy')
            edge_array = np.load(edge_path)
            print(f"edge type: {edge_array.dtype} Bytes: {edge_array.nbytes}, Shape: {edge_array.shape}")
            edge_array_size[0] = edge_array.nbytes
            edge_shape_list = np.array(edge_array.shape)

        #Sharing edge information
        self.comm_manager.local_comm.Barrier()
        self.comm_manager.local_comm.Bcast(edge_array_size, root = 0)

        self.comm_manager.local_comm.Barrier()
        self.comm_manager.local_comm.Bcast(edge_shape_list, root = 0)
        edge_shape = torch.Size(edge_shape_list)


        self.shared_UVA_manager = Shared_UVA_Tensor_Manager(self.comm_manager, "/shared_mem", edge_array_size[0])
        node_edges = self.shared_UVA_manager.get_tensor(dtype=cp.int64, tensor_shape=edge_shape)
        print(f"node edges shape: {node_edges.shape} device: {node_edges.device} array: {node_edges}" )
        self.shared_UVA_manager.write_np_array(node_edges, edge_array)
        print(f"After COPY node edges shape: {node_edges.shape} device: {node_edges.device} array: {node_edges}" )
        del edge_array

        gc.collect()
        print("Freed local graph data")



        label_path = osp.join(self.dir, 'raw', 'node_label.npy')
        node_labels = np.load(label_path)
        node_labels_torch =  torch.from_numpy(node_labels).to(torch.long).to(self.device)
        
        non_nan_indices = np.where(~np.isnan(node_labels))[0]
        non_nan_indices = torch.from_numpy(non_nan_indices)
        
       
        n_nodes = 111059956	
        self.graph = dgl.graph((node_edges[0,:],node_edges[1,:]), num_nodes=n_nodes)
        self.graph.ndata['label'] = node_labels_torch

        print(self.graph.formats())


        total_count = len(non_nan_indices)
        train_size = int(0.6 * total_count)
        val_size = int(0.2 * total_count)

        #print(f"Total count: {total_count}")

        train_indices = non_nan_indices[:train_size]
        val_indices = non_nan_indices[train_size:train_size + val_size]
        test_indices = non_nan_indices[train_size + val_size:]

        train_mask = torch.zeros(n_nodes, dtype=torch.bool).to(self.device)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool).to(self.device)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool).to(self.device)


        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True

        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return len(self.graphs)



class OGBDataset_Shared_CSC_UVA(DGLDataset):
    def __init__(self, args, comm_manager, device):
        self.dir = args.path
        self.size = args.dataset_size
        self.num_classes = args.num_classes
        self.args = args
        self.comm_manager = comm_manager
        self.device = device
        self.feat_data = None
        self.local_rank = comm_manager.local_rank
    #     super().__init__(name='IGB260M')
    # def process(self):

        if(self.args.feat_cpu):

            feat_array_size = np.empty([1], dtype="int")
            feat_shape_list = np.empty([2], dtype="int")
            emb = None
            if(self.local_rank == 0):
                print("Loading feature data into CPU for SSD simulation")
                feat_path = osp.join(self.dir, 'raw', 'node_feat.npy')
                emb = np.load(feat_path)
                feat_array_size[0] = emb.nbytes
                print(f"Feature data size: {emb.nbytes}B")
                feat_shape_list = np.array(emb.shape)

            #Sharing feat information
            self.comm_manager.local_comm.Barrier()
            self.comm_manager.local_comm.Bcast(feat_array_size, root = 0)

            self.comm_manager.local_comm.Barrier()
            self.comm_manager.local_comm.Bcast(feat_shape_list, root = 0)
            feat_shape = torch.Size(feat_shape_list)


            self.shared_UVA_manager_feat = Shared_UVA_Tensor_Manager(self.comm_manager, "/shared_mem_feat", feat_array_size[0])
            self.feat_data = self.shared_UVA_manager_feat.get_tensor(dtype=np.float32, tensor_shape=feat_shape, device=self.device)
            self.shared_UVA_manager_feat.write_np_array(self.feat_data, emb)
            #self.shared_UVA_manager_feat.write_np_array_gpu(self.feat_data, emb, self.device)

            del emb

        csc_indptr_array_size = np.empty([1], dtype="int")
        csc_indptr_shape_list = np.empty([1], dtype="int")
        csc_indptr_array = None

        csc_indices_array_size = np.empty([1], dtype="int")
        csc_indices_shape_list = np.empty([1], dtype="int")
        csc_indices_array = None

        csc_edge_ids_array_size = np.empty([1], dtype="int")
        csc_edge_ids_shape_list = np.empty([1], dtype="int")
        csc_edge_ids_array = None

        if(self.local_rank == 0):
            #(indptr, indices, edge_ids)
            indptr_path = osp.join(self.dir, 'raw', 'csc_indptr.npy')
            csc_indptr_array = np.load(indptr_path)
            print(f"edge type: {csc_indptr_array.dtype} Bytes: {csc_indptr_array.nbytes}, Shape: {csc_indptr_array.shape}")
            csc_indptr_array_size[0] = csc_indptr_array.nbytes
            csc_indptr_shape_list = np.array(csc_indptr_array.shape)

            indices_path = osp.join(self.dir, 'raw', 'csc_indices.npy')
            csc_indices_array = np.load(indices_path)
            print(f"edge type: {csc_indices_array.dtype} Bytes: {csc_indices_array.nbytes}, Shape: {csc_indices_array.shape}")
            csc_indices_array_size[0] = csc_indices_array.nbytes
            csc_indices_shape_list = np.array(csc_indices_array.shape)

            edge_ids_path = osp.join(self.dir, 'raw', 'csc_edge_ids.npy')
            csc_edge_ids_array = np.load(edge_ids_path)
            print(f"edge type: {csc_edge_ids_array.dtype} Bytes: {csc_edge_ids_array.nbytes}, Shape: {csc_edge_ids_array.shape}")
            csc_edge_ids_array_size[0] = csc_edge_ids_array.nbytes
            csc_edge_ids_shape_list = np.array(csc_edge_ids_array.shape)

        #Sharing edge information
        self.comm_manager.local_comm.Barrier()
        self.comm_manager.local_comm.Bcast(csc_indptr_array_size, root = 0)

        self.comm_manager.local_comm.Barrier()
        self.comm_manager.local_comm.Bcast(csc_indptr_shape_list, root = 0)
        csc_indptr_shape = torch.Size(csc_indptr_shape_list)

        self.comm_manager.local_comm.Barrier()
        self.comm_manager.local_comm.Bcast(csc_indices_array_size, root = 0)

        self.comm_manager.local_comm.Barrier()
        self.comm_manager.local_comm.Bcast(csc_indices_shape_list, root = 0)
        csc_indices_shape = torch.Size(csc_indices_shape_list)

        self.comm_manager.local_comm.Barrier()
        self.comm_manager.local_comm.Bcast(csc_edge_ids_array_size, root = 0)

        self.comm_manager.local_comm.Barrier()
        self.comm_manager.local_comm.Bcast(csc_edge_ids_shape_list, root = 0)
        csc_edge_ids_shape = torch.Size(csc_edge_ids_shape_list)


        self.shared_UVA_manager_indptr = Shared_UVA_Tensor_Manager(self.comm_manager, "/shared_mem_1", csc_indptr_array_size[0])
        csc_indptr = self.shared_UVA_manager_indptr.get_tensor(dtype=np.int64, tensor_shape=csc_indptr_shape, device=self.device)
        print(f"node edges shape: {csc_indptr.shape} device: {csc_indptr.device} array: {csc_indptr}" )
        self.shared_UVA_manager_indptr.write_np_array(csc_indptr, csc_indptr_array)
        print(f"After COPY node edges shape: {csc_indptr.shape} device: {csc_indptr.device} array: {csc_indptr}" )
        del csc_indptr_array

        self.shared_UVA_manager_indicies = Shared_UVA_Tensor_Manager(self.comm_manager, "/shared_mem_2", csc_indices_array_size[0])
        csc_indicies = self.shared_UVA_manager_indicies.get_tensor(dtype=np.int64, tensor_shape=csc_indices_shape, device=self.device)
        print(f"node edges shape: {csc_indicies.shape} device: {csc_indicies.device} array: {csc_indicies}" )
        self.shared_UVA_manager_indicies.write_np_array(csc_indicies, csc_indices_array)
        print(f"After COPY node edges shape: {csc_indicies.shape} device: {csc_indicies.device} array: {csc_indicies}" )
        del csc_indices_array

        self.shared_UVA_manager_edge_id = Shared_UVA_Tensor_Manager(self.comm_manager, "/shared_mem_3", csc_edge_ids_array_size[0])
        edge_ids_indicies = self.shared_UVA_manager_edge_id.get_tensor(dtype=np.int64, tensor_shape=csc_edge_ids_shape, device=self.device)
        print(f"node edges shape: {edge_ids_indicies.shape} device: {edge_ids_indicies.device} array: {edge_ids_indicies}" )
        self.shared_UVA_manager_edge_id.write_np_array(edge_ids_indicies, csc_edge_ids_array)
        print(f"After COPY node edges shape: {edge_ids_indicies.shape} device: {edge_ids_indicies.device} array: {edge_ids_indicies}" )
        del csc_edge_ids_array


        gc.collect()
        print("Freed local graph data")



        label_path = osp.join(self.dir, 'raw', 'node_label.npy')
        node_labels = np.load(label_path)
        node_labels_torch =  torch.from_numpy(node_labels).to(torch.long).to(self.device)
        
        non_nan_indices = np.where(~np.isnan(node_labels))[0]
        non_nan_indices = torch.from_numpy(non_nan_indices)
        
       
        n_nodes = 111059956	
        self.graph = dgl.graph(('csc',(csc_indptr, csc_indicies, edge_ids_indicies)), num_nodes=n_nodes, idtype=torch.int64, device=self.device)
        self.graph  = self.graph.formats('csc')
        self.graph.ndata['label'] = node_labels_torch
        print(self.graph.formats())


        total_count = len(non_nan_indices)
        train_size = int(0.6 * total_count)
        val_size = int(0.2 * total_count)

        #print(f"Total count: {total_count}")

        train_indices = non_nan_indices[:train_size]
        val_indices = non_nan_indices[train_size:train_size + val_size]
        test_indices = non_nan_indices[train_size + val_size:]

        train_mask = torch.zeros(n_nodes, dtype=torch.bool).to(self.device)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool).to(self.device)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool).to(self.device)


        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True

        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

        
    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return len(self.graphs)


