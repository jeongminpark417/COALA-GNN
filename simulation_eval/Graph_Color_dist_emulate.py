import time
import argparse, datetime

import dgl
import dgl.function as fn

import numpy as np
import torch

from GIDS import ID_Loader

#from custom_neighbor import sample_neighbors2

from dataloader import IGB260MDGLDataset, OGBDGLDataset, load_ogb

#from custom_sampler import NeighborSampler2

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
                graph_dict[node_id] = 0

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
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)

    parser.add_argument('--dist', type=str, default='baseline')
    parser.add_argument('--num_parts', type=int, default=2)
    parser.add_argument('--cache_percent', type=float, default=1.0)



    parser.add_argument('--color_path', type=str, default='./',
    help='path containing the datasets')
    parser.add_argument('--eviction_policy', type=int, default=0)



    args = parser.parse_args()
    

    device = f'cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu'
    print("device: ", device)
    if(args.data == 'IGB'):
        print("Dataset: IGB")
        dataset = IGB260MDGLDataset(args)
        g = dataset[0]

    elif(args.data == 'OGB'):
        print("Dataset: OGB")
        # dataset = OGBDGLDataset(args)
        # g = dataset[0]
        # g  = g.formats('csc')

        g, labels = load_ogb("ogbn-papers100M", args.path)
    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']

    
  #  lib = ctypes.CDLL('/root/Distributed-GIDS/multi_node_evaluation/custom_sampling/libexample.so', mode=ctypes.RTLD_GLOBAL)
#    sampler = NeighborSampler2([int(fanout) for fanout in args.fan_out.split(',')], hot_nodes=hot_nodes_tensor)
    sampler = dgl.dataloading.MultiLayerNeighborSampler([int(fanout) for fanout in args.fan_out.split(',')])
    
    train_nid = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
    val_nid = torch.nonzero(g.ndata['val_mask'], as_tuple=True)[0]
    test_nid = torch.nonzero(g.ndata['test_mask'], as_tuple=True)[0]

    in_feats = g.ndata['features'].shape[1]

    # patrition_file = '/mnt/nvme15/adj/partition/ogbn_papers100M.adj/part_0'
    # partition_dict = read_graph_file(patrition_file)
    # print("train nid: ", train_nid)
    # print("original train nid size: ", len(train_nid))

    # filtered_train_nid = torch.tensor([nid for nid in train_nid if partition_dict.get(nid.item(), -1) == 0])
    # print("filtered train nid: ", filtered_train_nid)
    # print("filtered train nid size: ", len(filtered_train_nid))

    #train_nid = filtered_train_nid

    num_nodes = g.number_of_nodes()

    color_tensor = torch.load(args.color_path + "color.pt")
    color_topk = torch.load(args.color_path + "topk.pt")
    static_info_tensor = torch.load(args.color_path + "static_info.pt")

    num_ways = 32
    num_sets = int(num_nodes * args.cache_percent / args.num_parts / num_ways)
    print("Number of Sets: ", num_sets)

    dim = in_feats
    train_dataloader = ID_Loader(
        g,
        train_nid,
        sampler,
        batch_size=args.batch_size * args.num_parts,
        shuffle=True,
        dim = dim,
        device=device,
        num_sets = num_sets,
        #num_sets = 16,
        num_ways = num_ways,
        color_tensor = color_tensor,
        color_topk = color_topk,
        num_gpus = args.num_parts,
        distribute_method = args.dist,
        static_info_tensor = static_info_tensor,
        eviction_policy = args.eviction_policy
    )

    access_count_tensor = torch.zeros(num_nodes)
    print("num nodes: ", num_nodes)
    print("Dim: ", dim)


    dictionary_time = 0
    print("start training")
    train_start = time.time()

    for i in range(2):
        num_accesses = 0
        for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):
            if(step % 100 == 0):
                print("step: ", step)
            #     break
            num_accesses += len(input_nodes)
            # for node_tensor in input_nodes:
            #     node = node_tensor.item()
            #     #print("node: ", node)
            #     dict_start = time.time()
            #     if(node in partition_dict):
            #         in_partition += 1

            #         access_count_tensor[node] += 1
            #         #print("in partition")
            #     else:
            #         out_partition += 1
            #         access_count_tensor[node] += 1
            #         #print("not in partition")
            #     dict_end = time.time()
            #     dictionary_time += (dict_end - dict_start)
            # if(step <= 3):
            #     print("step: ", step)
            # break
        print("Print Counters")
        train_dataloader.print_counters()
        print("num accesses: ", num_accesses)

