import argparse, datetime
import dgl
import sklearn.metrics
import torch, torch.nn as nn, torch.optim as optim
import time, tqdm, numpy as np
from models import *
from GIDS_dataloader import IGB260MDGLDataset, OGBDGLDataset

from GIDS_dataloader import IGBHeteroDGLDataset2

import csv 
import warnings

import torch.cuda.nvtx as t_nvtx
import nvtx
import threading
import gc

import GIDS
from GIDS import GIDS_DGLDataLoader, ID_Loader

from ogb.graphproppred import DglGraphPropPredDataset
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

import random


torch.manual_seed(0)
dgl.seed(0)
warnings.filterwarnings("ignore")



def find_closest_hot_node(g, node_ids, hot_nodes, k, device):
  
    hot_nodes = hot_nodes.to(device)  # Ensure the hot_nodes tensor is on GPU
    
    closest_hot_nodes = torch.full((node_ids,), -1, dtype=torch.int64, device=device)
    
    counter = 0
    for i in range(node_ids):
        if(i%10000 == 0):
            print("iter: ", i)
        node_id = i
        visited = set([node_id])
        queue = [node_id]
        current_hop = 0
        
        while queue and current_hop <= k:
            current_level_size = len(queue)
            current_level_hot_nodes = []
            
            for _ in range(current_level_size):
                current_node = queue.pop(0)
                
               # print("hont node vale for ",current_node, " is ", hot_nodes[current_node].item())
                if (hot_nodes[current_node].item() <= 15):
                   # print("append")
                    current_level_hot_nodes.append(current_node)
                
                for neighbor in g.predecessors(current_node).tolist():
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            if current_level_hot_nodes:
                # Pick a random hot node from the current level hot nodes 
                closest_hot_nodes[i] = random.choice(current_level_hot_nodes)
                counter += 1
                break
            
            current_hop += 1
    
    print("id with label: ", counter)
    print("percentage: ", counter / len(closest_hot_nodes) * 100)
    return closest_hot_nodes
    
def track_acc_GIDS(g,  args, device, num_training_nodes, hot_nodes, label_array=None, key_offset=None):

    
    closest_hot_nodes = find_closest_hot_node(g, num_training_nodes, hot_nodes, 2, device).cpu()
    print(closest_hot_nodes)
  
    out_file = args.out_file
    with open(out_file, 'wb') as f:
        f.write(closest_hot_nodes.numpy().tobytes())

    return
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
               [int(fanout) for fanout in args.fan_out.split(',')]
               )


    train_nid = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
    val_nid = torch.nonzero(g.ndata['val_mask'], as_tuple=True)[0]
    test_nid = torch.nonzero(g.ndata['test_mask'], as_tuple=True)[0]
  
    dim = args.emb_size
    in_feats = dim
    
    if args.model_type == 'gcn':
        model = GCN(in_feats, args.hidden_channels, args.num_classes, 
            args.num_layers).to(device)
    if args.model_type == 'sage':
        model = SAGE(in_feats, args.hidden_channels, args.num_classes, 
            args.num_layers).to(device)
    if args.model_type == 'gat':
        model = GAT(in_feats, args.hidden_channels, args.num_classes, 
            args.num_layers, args.num_heads).to(device)
    
    
    train_dataloader =  ID_Loader (
        g,
        train_nid,
        sampler,
        args.batch_size,
        dim,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        use_alternate_streams=False
    )

    loss_fcn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.learning_rate, weight_decay=args.decay
        )

    warm_up_iter = 1000
    # Setup is Done
    for epoch in tqdm.tqdm(range(args.epochs)):
        epoch_start = time.time()
        epoch_loss = 0
        train_acc = 0
        model.train()

        batch_input_time = 0
        train_time = 0
        transfer_time = 0
        e2e_time = 0
        e2e_time_start = time.time()
        id_list = []

        for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
            
            print("input nodes: ", input_nodes)
            divisible_by_8 = torch.sum(input_nodes % 8 == 0)

            # Calculate the percentage
            percentage = (divisible_by_8.item() / input_nodes.numel()) * 100


            print("Number of elements divisible by 8:", divisible_by_8.item())
            print("Percentage of elements divisible by 8:", percentage)

            if(step == 3):
                break

    
       

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Loading dataset
    parser.add_argument('--path', type=str, default='/mnt/nvme14/IGB260M', 
        help='path containing the datasets')
    parser.add_argument('--out_file', type=str, default='./out_file.out')
    parser.add_argument('--dataset_size', type=str, default='experimental',
        choices=['experimental', 'small', 'medium', 'large', 'full'], 
        help='size of the datasets')
    parser.add_argument('--num_classes', type=int, default=19, 
        choices=[19, 2983, 172, 348,349, 350, 153, 152], help='number of classes')
    parser.add_argument('--in_memory', type=int, default=0, 
        choices=[0, 1], help='0:read only mmap_mode=r, 1:load into memory')
    parser.add_argument('--synthetic', type=int, default=0,
        choices=[0, 1], help='0:nlp-node embeddings, 1:random')
    parser.add_argument('--data', type=str, default='IGB')
    parser.add_argument('--emb_size', type=int, default=1024)
    
    # Model
    parser.add_argument('--model_type', type=str, default='gcn',
                        choices=['rgat', 'rsage', 'rgcn'])
    parser.add_argument('--modelpath', type=str, default='deletethis.pt')
    parser.add_argument('--model_save', type=int, default=0)

    # Model parameters 
    parser.add_argument('--fan_out', type=str, default='10,15')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--decay', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--log_every', type=int, default=2)

    #GIDS parameter
    parser.add_argument('--GIDS', action='store_true', help='Enable GIDS Dataloader')
    parser.add_argument('--num_ssd', type=int, default=1)
    parser.add_argument('--cache_size', type=int, default=8)
    parser.add_argument('--uva', type=int, default=0)
    parser.add_argument('--uva_graph', type=int, default=0)
    parser.add_argument('--wb_size', type=int, default=6)

    parser.add_argument('--device', type=int, default=0)

    #GIDS Optimization
    parser.add_argument('--accumulator', action='store_true', help='Enable Storage Access Accmulator')
    parser.add_argument('--bw', type=float, default=5.8, help='SSD peak bandwidth in GB/s')
    parser.add_argument('--l_ssd', type=float, default=11.0, help='SSD latency in microseconds')
    parser.add_argument('--l_system', type=float, default=20.0, help='System latency in microseconds')
    parser.add_argument('--peak_percent', type=float, default=0.95)

    parser.add_argument('--num_iter', type=int, default=1)

    parser.add_argument('--cpu_buffer', action='store_true', help='Enable CPU Feature Buffer')
    parser.add_argument('--cpu_buffer_percent', type=float, default=0.2, help='CPU feature buffer size (0.1 for 10%)')
    parser.add_argument('--pin_file', type=str, default="/mnt/nvme16/pr_full.pt", 
        help='Pytorch Tensor File for the list of nodes that will be pinned in the CPU feature buffer')

    parser.add_argument('--window_buffer', action='store_true', help='Enable Window Buffering')



    #GPU Software Cache Parameters
    parser.add_argument('--page_size', type=int, default=8)
    parser.add_argument('--offset', type=int, default=0, help='Offset for the feature data stored in the SSD') 
    parser.add_argument('--num_ele', type=int, default=100, help='Number of elements in the dataset (Total Size / sizeof(Type)') 
    parser.add_argument('--cache_dim', type=int, default=1024) #CHECK
    parser.add_argument('--shared', type=int, default=0, help='Number of elements in the dataset (Total Size / sizeof(Type)')


    args = parser.parse_args()
    print("GIDS DataLoader Setting")
    print("GIDS: ", args.GIDS)
    print("CPU Feature Buffer: ", args.cpu_buffer)
    print("Window Buffering: ", args.window_buffer)
    print("Storage Access Accumulator: ", args.accumulator)

    labels = None
    key_offset = None

    device = f'cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu'
    if(args.data == 'IGB'):
        print("Dataset: IGB")
        dataset = IGB260MDGLDataset(args)
        g = dataset[0]
    elif(args.data == "OGB"):
        print("Dataset: OGB")
        dataset = OGBHeteroDGLDatasetMassive(args)
        g = dataset[0]
        g = g.formats('csc')
    else:
        g=None
        dataset=None
    
  
    num_training_nodes = dataset.n_labeled_idx
    print("num_training_nodes: ", num_training_nodes)
    
    file_path = '/mnt/raid0/norm_pr_small_8bit.pt'

    # Step 2: Read the binary data into a NumPy array
    data_np = np.fromfile(file_path, dtype=np.uint8)

    # Step 3: Convert the NumPy array into a PyTorch tensor
    hot_nodes = torch.from_numpy(data_np).to(device)



    track_acc_GIDS(g,  args, device, num_training_nodes, hot_nodes, labels)
    #track_acc(g, args, device, labels)




