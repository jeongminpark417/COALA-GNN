import sklearn.metrics
import time
import argparse, datetime
import torch, torch.nn as nn, torch.optim as optim
import os

import dgl
import dgl.function as fn

import numpy as np
import torch
from models import *
import tqdm
import sys

#import GIDS
from Shared_Tensor import Shared_UVA_Tensor_Manager

from ssd_gnn_dataloader import IGBDatast_Shared_UVA, IGBDatast_Shared_CSC_UVA, OGBDataset_Shared_UVA, OGBDataset_Shared_CSC_UVA


import ctypes
import mpi4py
from mpi4py import MPI
import torch.distributed as dist

from Shared_Tensor import Shared_UVA_Tensor_Manager, MPI_Comm_Manager, Node_Distributor
from Shared_Tensor import SSD_INFO, SSD_GNN_DataLoader




def read_graph_file(filename):
    # Dictionary to store the node_id as key and 1 as value
    graph_dict = {}

    # Open the file and read line by line
    with open(filename, 'r') as file:
        for line in file:
            # Split the line into parts
            parts = line.split()
            if parts:
                # The first part is the node_id
                node_id = int(parts[0])
                # Set the value 1 for each node_id in the dictionary
                graph_dict[node_id] = 0

    return graph_dict


def train( g, args, device, Comm_Manager, dim, page_size, num_classes, feat_shared_uva = None):
    if(feat_shared_uva != None):
        print("SSD_GNN simulation train")
    else:
        print("SSD_GNN real SSD train")

    if(args.data == 'IGB'):
        meta_path = args.path + str(args.dataset_size) + "/"
    elif (args.data == 'OGB'):
        meta_path = args.path
    else:
        meta_path = None
    color_file = meta_path + "color.npy"
    topk_file = meta_path + "topk.npy"
    score_file = meta_path + "score.npy"



    train_nid = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0].clone()
    test_nid = torch.nonzero(g.ndata['test_mask'], as_tuple=True)[0].clone()
    print(f"train nid len: {(train_nid.shape)}")
    Node_Distributor_Manager = Node_Distributor(Comm_Manager, train_nid, args.batch_size, color_file, topk_file, score_file)

    sampler = dgl.dataloading.MultiLayerNeighborSampler(
               [int(fanout) for fanout in args.fan_out.split(',')]
               )

    cache_size = args.cache_size

    fan_out = [int(fanout) for fanout in args.fan_out.split(',')]

    SSD_manager = SSD_INFO(num_ssds = 1, 
                           page_size = page_size,
                           num_elems = 1024, 
                           ssd_read_offset = 0)

    train_loader = SSD_GNN_DataLoader(
                        SSD_manager,
                        Node_Distributor_Manager, 
                        g,
                        sampler,
                        args.batch_size, 
                        dim,
                        fan_out,
                        cache_size,
                        device,
                        cache_backend = "nvshmem",
                        sim_buf = feat_shared_uva,
                        shuffle = False)



    print("train dataloader init done")

    if args.model_type == 'gcn':
        model = GCN(dim, args.hidden_channels, num_classes, 
            args.num_layers).to(device)
    if args.model_type == 'sage':
        model = SAGE(dim, args.hidden_channels, num_classes, 
            args.num_layers).to(device)
        # model = DistSAGE(dim, args.hidden_channels, num_classes, args.num_layers, torch.nn.functional.relu
        #     ).to(device)
    if args.model_type == 'gat':
        model = GAT(dim, args.hidden_channels, num_classes, 
            args.num_layers,  args.num_heads).to(device)


    model = torch.nn.parallel.DistributedDataParallel(model)

    loss_fcn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.learning_rate, weight_decay=args.decay
        )


    #for step, (blocks) in enumerate(train_loader):
    count = 0
    num_sampled_nodes = 0
    print("Training start")
    model.train()
    for epoch in range(args.epochs):
        print(f"Epoch: {epoch}")
        epoch_start = time.time()
        for step, (input_nodes, seeds, blocks, fetch_feature) in enumerate(train_loader):
            num_sampled_nodes += len(input_nodes)

            if(step % 100 == 0):
                print(f"Rank: {Comm_Manager.local_rank} step: {step}")


            count += 1
            
            batch_labels = blocks[-1].dstdata['labels']
            blocks = [block.int().to(device) for block in blocks]
            batch_labels = batch_labels.view(-1).to(device)
            batch_pred = model(blocks, fetch_feature)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_end = time.time()
        print(f"Epoch Time: {epoch_end - epoch_start}")
        print(f"Total number of iterations: {count}")
        print(f"Number of sampled nodes : {num_sampled_nodes}")
        train_loader.print_stats()

    Test_Node_Distributor_Manager = Node_Distributor(Comm_Manager, test_nid, args.batch_size, color_file, topk_file, score_file)
    test_loader = SSD_GNN_DataLoader(
                    SSD_manager,
                    Test_Node_Distributor_Manager, 
                    g,
                    sampler,
                    args.batch_size, 
                    dim,
                    fan_out,
                    cache_size,
                    device,
                    cache_backend = "nvshmem",
                    sim_buf = feat_shared_uva,
                    shuffle = False)

    
    model.eval()
    predictions = []
    labels = []
    g.ndata['labels'] = g.ndata['label']
    count = 0
    with torch.no_grad():
        #for _, _, blocks in test_dataloader:
        for step, (input_nodes, seeds, blocks, fetch_feature) in enumerate(test_loader):
            if(count % 100 == 0):
                print("Eval step: ", count)
      
            blocks = [block.to(device) for block in blocks]
            batch_labels = blocks[-1].dstdata['labels']
            batch_labels = batch_labels.view(-1)
            labels.append(blocks[-1].dstdata['label'].cpu().numpy())
            predictions.append(model(blocks, fetch_feature).argmax(1).cpu().numpy())
            count += 1
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        test_acc = sklearn.metrics.accuracy_score(labels, predictions)*100
    print("Test Acc {:.2f}%".format(test_acc))


            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/mnt/nvme14/IGB260M',
        help='path containing the datasets')
    parser.add_argument('--dataset_size', type=str, default='experimental',
        choices=['experimental', 'small', 'medium', 'large', 'full'],
        help='size of the datasets')
    parser.add_argument('--num_classes', type=int, default=19,
        choices=[19, 2983, 172, 128], help='number of classes')
    parser.add_argument('--in_memory', type=int, default=0,
        choices=[0, 1], help='0:read only mmap_mode=r, 1:load into memory')
    parser.add_argument('--synthetic', type=int, default=0,
        choices=[0, 1], help='0:nlp-node embeddings, 1:random')
    parser.add_argument('--data', type=str, default='IGB')
    parser.add_argument('--emb_size', type=int, default=1024)
    parser.add_argument('--shared', type=int, default=1,
        choices=[0, 1], help='0:copy graph=r, 1:shared memory')

    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--uva', type=int, default=0)
    parser.add_argument('--uva_graph', type=int, default=0)

    parser.add_argument('--model_type', type=str, default='sage',
                        choices=['gat', 'sage', 'gcn'])

    parser.add_argument('--fan_out', type=str, default='10,5,5')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--decay', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--log_every', type=int, default=2)

    
    parser.add_argument('--page_size', type=int, default=8)
    parser.add_argument('--offset', type=int, default=0, help='Offset for the feature data stored in the SSD') 
    parser.add_argument('--num_ele', type=int, default=100, help='Number of elements in the dataset (Total Size / sizeof(Type)') 
    parser.add_argument('--cache_dim', type=int, default=1024) #CHECK
    parser.add_argument('--num_ssd', type=int, default=1) 
    parser.add_argument('--cache_size', type=int, default=1024) 



    parser.add_argument('--eviction_policy', type=int, default=0)

    parser.add_argument('--feat_cpu', action='store_true', help='Store features on CPU')




    args = parser.parse_args()
    labels = None

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(15385)


    in_feats = args.cache_dim

    if not MPI.Is_initialized():
        print("MPI init")
        MPI.Init()

    Comm_Manager = MPI_Comm_Manager()
    device = 'cuda:' + str(Comm_Manager.local_rank)
    print("device: ", device)
    torch.cuda.set_device(device)

    Comm_Manager.initialize_nested_process_group()


    if(args.data == 'IGB'):
        print("Dataset: IGB")
        dataset = IGBDatast_Shared_CSC_UVA(args, Comm_Manager, device)        
        g = dataset[0]
        dim = 1024
        page_size = 4096
        num_classes = args.num_classes
    elif(args.data == 'OGB'):
        print("Dataset: OGB")
        dataset = OGBDataset_Shared_CSC_UVA(args, Comm_Manager, device)        
        g = dataset[0]
        dim = 128
        page_size = 512
        num_classes = 172
    else:
        print("Unsupported Dataset")
        # dataset = OGBDGLDataset(args)
        # g = dataset[0]
        # g  = g.formats('csc')
        sys.exit(1)
    g.ndata['labels'] = g.ndata['label']

    

    print(f"RANK = {os.environ.get('RANK')}")
    print(f"WORLD_SIZE = {os.environ.get('WORLD_SIZE')}")
    print(f"MASTER_ADDR = {os.environ.get('MASTER_ADDR')}")
    print(f"MASTER_PORT = {os.environ.get('MASTER_PORT')}")

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    feat_data = None
    if(args.feat_cpu):
        feat_data = dataset.feat_data

#    feat_data = torch.Tensor([1,2])
    train( g, args, device, Comm_Manager, dim, page_size, num_classes, feat_data)


    Comm_Manager.destroy_process_group()
    MPI.Finalize()

