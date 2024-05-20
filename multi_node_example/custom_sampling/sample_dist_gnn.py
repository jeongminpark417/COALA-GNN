import time
import argparse, datetime

import dgl
import dgl.function as fn

import numpy as np
import torch

from GIDS import ID_Loader

from custom_neighbor import sample_neighbors2

from dataloader import IGB260MDGLDataset, OGBDGLDataset

from custom_sampler import NeighborSampler2

import ctypes

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
                graph_dict[node_id] = 1

    return graph_dict


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

    parser.add_argument('--fan_out', type=str, default='5,5')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=0)


    args = parser.parse_args()
    

    device = f'cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu'
    print("device: ", device)
    if(args.data == 'IGB'):
        print("Dataset: IGB")
        dataset = IGB260MDGLDataset(args)
        g = dataset[0]

    elif(args.data == 'OGB'):
        dataset = OGBDGLDataset(args)
        g = dataset[0]
    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']

    
    lib = ctypes.CDLL('/root/Distributed-GIDS/multi_node_evaluation/custom_sampling/libexample.so', mode=ctypes.RTLD_GLOBAL)
#    sampler = NeighborSampler2([int(fanout) for fanout in args.fan_out.split(',')], hot_nodes=hot_nodes_tensor)
    sampler = dgl.dataloading.MultiLayerNeighborSampler([int(fanout) for fanout in args.fan_out.split(',')])
    
    train_nid = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
    val_nid = torch.nonzero(g.ndata['val_mask'], as_tuple=True)[0]
    test_nid = torch.nonzero(g.ndata['test_mask'], as_tuple=True)[0]

    in_feats = g.ndata['features'].shape[1]

    patrition_file = '/mnt/nvme15/adj/partition/ogbn_papers100M.adj/part_0'
    partition_dict = read_graph_file(patrition_file)
    print("train nid: ", train_nid)

    dim = in_feats
    train_dataloader = ID_Loader(
        g,
        train_nid,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        dim = dim

    )

    patrition_file = '/mnt/nvme15/adj/partition/ogbn_papers100M.adj/part_0'
    partition_dict = read_graph_file(patrition_file)
    
    in_partition = 0
    out_partition = 0

    for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):

        for node_tensor in input_nodes:
            node = node_tensor.item()
            #print("node: ", node)
            if(node in partition_dict):
                in_partition += 1
                #print("in partition")
            else:
                out_partition += 1
                #print("not in partition")

        break

    print("Partition Percentage: ", (in_partition/ (in_partition+out_partition)))

        #if(step == 100):
        #    print("step 100\n")
        #    break
