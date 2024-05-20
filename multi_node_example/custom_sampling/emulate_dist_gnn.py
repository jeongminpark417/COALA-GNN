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
    print("original train nid size: ", len(train_nid))

    filtered_train_nid = torch.tensor([nid for nid in train_nid if partition_dict.get(nid.item(), -1) == 0])
    print("filtered train nid: ", filtered_train_nid)
    print("filtered train nid size: ", len(filtered_train_nid))

    train_nid = filtered_train_nid
    dim = in_feats
    train_dataloader = ID_Loader(
        g,
        train_nid,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        dim = dim

    )

    dict_create_start = time.time()

    patrition_file = '/mnt/nvme15/adj/partition/ogbn_papers100M.adj/part_0'
    partition_dict = read_graph_file(patrition_file)
    dict_create_end = time.time()
    print("dict creation time: ", dict_create_end - dict_create_start)

    in_partition = 0
    out_partition = 0

    num_nodes = g.number_of_nodes()
    access_count_tensor = torch.zeros(num_nodes)
    print("num nodes: ", num_nodes)
    


    dictionary_time = 0
    print("start training")
    train_start = time.time()
    for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):

#        if(step == 2):
#            break

        for node_tensor in input_nodes:
            node = node_tensor.item()
            #print("node: ", node)
            dict_start = time.time()
            if(node in partition_dict):
                in_partition += 1

                access_count_tensor[node] += 1
                #print("in partition")
            else:
                out_partition += 1
                access_count_tensor[node] += 1
                #print("not in partition")
            dict_end = time.time()
            dictionary_time += (dict_end - dict_start)

        # break
    train_end = time.time()
    
    percentages = [10, 25, 50, 75, 90]

    print("training time: ", train_end - train_start)
    print("dictonary time: ", dictionary_time)

    print("in_partition count: ", in_partition)
    print("Partition Percentage: ", (in_partition/ (in_partition+out_partition)))

    dict_list = list(partition_dict.keys())
    indices_tensor = torch.tensor(dict_list)
    filtered_tensor = access_count_tensor[indices_tensor]

    min_val = torch.min(filtered_tensor)
    max_val = torch.max(filtered_tensor)
    std_dev = torch.std(filtered_tensor)
    average = torch.mean(filtered_tensor)
    median  = torch.median(filtered_tensor)
    # Zero indices
    zero_indices = torch.where(filtered_tensor <= 1)[0]

    # Output results
    print(f"Filtered Tensor: {filtered_tensor}")
    print(f"Filtered Tensor len: {len(filtered_tensor)}")
    print(f"Number Partitoned Nodes: {len(dict_list)}")
    print(f"Minimum Value: {min_val}")
    print(f"Maximum Value: {max_val}")
    print(f"Standard Deviation: {std_dev}")
    print(f"Average (Mean): {average}")
    print(f"Average (Median): {median}")
#    print(f"Indices with One Values: {zero_indices}")
    print(f"Less than once accesses: {len(zero_indices)}")
   
    total_accesses = filtered_tensor.sum()
    thresholds = [total_accesses * (p / 100.0) for p in percentages]
    sorted_access_counts = torch.sort(filtered_tensor, descending=True)[0]
    cumulative_sums = torch.cumsum(sorted_access_counts, dim=0)

    print(f"Local total accesses: {total_accesses}")
    threshold_indices = [torch.searchsorted(cumulative_sums, threshold) for threshold in thresholds]
    for p, idx in zip(percentages, threshold_indices):
        print(f"Local - Indices needed to surpass {p}% of total accesses: {idx.item() + 1}")




    # outside of partition
    full_mask = torch.zeros(num_nodes, dtype=torch.bool)
    full_mask[indices_tensor] = True
    mask_not_in_dict_list = ~full_mask

    # Filter out zeros
    mask_non_zero = access_count_tensor != 0
    combined_mask = mask_not_in_dict_list & mask_non_zero
    second_filtered_tensor = access_count_tensor[combined_mask]

    min_val = torch.min(second_filtered_tensor)
    max_val = torch.max(second_filtered_tensor)
    std_dev = torch.std(second_filtered_tensor)
    average = torch.mean(second_filtered_tensor)
    median = torch.median(second_filtered_tensor)
    # Find indices with values less than or equal to 1
    zero_indices = torch.where(second_filtered_tensor <= 1)[0]

    print("inout_partition count: ", out_partition)
    # Output results for second_filtered_tensor
    print(f"Second Filtered Tensor: {second_filtered_tensor}")
    print(f"Length of Second Filtered Tensor: {len(second_filtered_tensor)}")
    print(f"Number of Partitioned Nodes: {len(dict_list)}")  # Number of indices from the dict_list
    print(f"Minimum Value: {min_val}")
    print(f"Maximum Value: {max_val}")
    print(f"Standard Deviation: {std_dev}")
    print(f"Average (Mean): {average}")
    print(f"Median: {median}")
    print(f"Count of values <= 1: {len(zero_indices)}") 


    remote_total_accesses = second_filtered_tensor.sum()
    sorted_remote_access_counts = torch.sort(second_filtered_tensor, descending=True)[0]
    cumulative_remote_sums = torch.cumsum(sorted_remote_access_counts, dim=0)

    print(f"Remote total accesses: {remote_total_accesses}")
    remote_thresholds = [remote_total_accesses * (p / 100.0) for p in percentages]

    remote_threshold_indices = [torch.searchsorted(cumulative_remote_sums, threshold) for threshold in remote_thresholds]
    for p, idx in zip(percentages, remote_threshold_indices):
        print(f"Remote - Indices needed to surpass {p}% of total accesses: {idx.item() + 1}")

