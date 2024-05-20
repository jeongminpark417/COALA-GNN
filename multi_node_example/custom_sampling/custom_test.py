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

   

    with open('/mnt/raid0/IGB_small_closest.pt', 'rb') as f:
        hot_nodes = np.fromfile(f, dtype=np.int64)

    print("hot ndoes shape: ", hot_nodes.shape)

    hot_nodes = hot_nodes.reshape(g.ndata['features'].shape[0])  # Uncomment if you know the shape

    hot_nodes_tensor = torch.from_numpy(hot_nodes).pin_memory()

    #sampler = dgl.dataloading.NeighborSampler([int(fanout) for fanout in args.fan_out.split(',')], replace=True)
    sampler = NeighborSampler2([int(fanout) for fanout in args.fan_out.split(',')], hot_nodes=hot_nodes_tensor)

    
    train_nid = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
    val_nid = torch.nonzero(g.ndata['val_mask'], as_tuple=True)[0]
    test_nid = torch.nonzero(g.ndata['test_mask'], as_tuple=True)[0]

    in_feats = g.ndata['features'].shape[1]

    torch.manual_seed(42)


    dim = in_feats
#    train_dataloader = dgl.dataloading.DataLoader(
    train_dataloader = ID_Loader(
        g,
        train_nid,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        dim = dim

    )

    counts_dict = {}

    sorted_counts = None
    sorted_unique_elements_by_count = None
    distribution_set = None

    sample_node_metadata = torch.zeros( (g.ndata['features'].shape[0]), dtype=torch.bool, device=device)
    base_node_data = torch.zeros( (g.ndata['features'].shape[0]), dtype=torch.bool, device=device)

    distinct_hit = torch.zeros( (g.ndata['features'].shape[0]), dtype=torch.bool, device=device)
    distinct_hit2 = torch.zeros( (g.ndata['features'].shape[0]), dtype=torch.bool, device=device)

    hit_cnt = 0
    miss_cnt = 0

    baseline_hit_cnt = 0

    total_cnt = 0
    mask_cnt = 0

    warm_up_iter = 1000
    for step, (input_nodes, seeds, blocks) in enumerate(train_dataloader):

        if(step == 0):
            shuffled_seed_nodes = train_dataloader.dataset._indices[warm_up_iter:(warm_up_iter+100)*args.batch_size]
            cur_hot_nodes = hot_nodes_tensor[shuffled_seed_nodes].to(device)
            cur_unique_elements, cur_counts = torch.unique(cur_hot_nodes, return_counts=True)

            sorted_indices = torch.argsort(cur_counts, descending=True)
            sorted_unique_elements_by_count = (cur_unique_elements[sorted_indices])[1:(150*2)]
            sorted_counts = (cur_counts[sorted_indices])[1:100]
            #distribution_set = set(sorted_unique_elements_by_count.cpu().tolist())


            print("Unique elements size:", len(sorted_unique_elements_by_count))
            print("Counts:", sorted_counts)
           # print("Set:", distribution_set)


        if(step < warm_up_iter):

            cur_hot_nodes = blocks[0].hot_nodes
            mask = (cur_hot_nodes[:, None] == sorted_unique_elements_by_count).any(dim=1)
            sample_node_metadata[input_nodes[mask]] = True
            base_node_data[input_nodes] = True


            # if(step == 0):
            #     print("cur hot nodes: ", cur_hot_nodes)
            #     print("sorted_unique_elements_by_count: ", sorted_unique_elements_by_count)
            #     print("mask size ", mask.sum())
            #     print("mask: ", mask)
            #print("baseline: ", base_node_data)
            # unique_elements, counts = torch.unique(blocks[0].hot_nodes, return_counts=True)

            # if(hot_nodes)

        #Testing
        elif(step < (warm_up_iter+100)):
            total_cnt += len(input_nodes)
            mask_cnt += len(input_nodes)
            true_count = sample_node_metadata[input_nodes].sum()
            hit_cnt += true_count.item()

            base_true_count = base_node_data[input_nodes].sum()
            baseline_hit_cnt += base_true_count.item()

            distinct_hit[input_nodes] = True


            # cur_hot_nodes = blocks[0].hot_nodes
            # mask = (cur_hot_nodes[:, None] == sorted_unique_elements_by_count).any(dim=1)
            # true_count = sample_node_metadata[input_nodes[mask]].sum()
            # hit_cnt += true_count.item()
            # mask_cnt += mask.sum().item()

            # print("mask size ", mask.sum())
            # print("input len size: ", len(input_nodes))

        
        elif(step == (warm_up_iter+100)):
            print("Hit count: ", hit_cnt)
            print("Baseline Hit count: ", baseline_hit_cnt)

            print("total mask count: ", mask_cnt)

            print("total count: ", total_cnt)
            print("Hit ratio: ", hit_cnt / mask_cnt)
            print("Baseline Hit ratio: ", baseline_hit_cnt / total_cnt)

            print("Cached Elemetns: ",  sample_node_metadata.sum())
            print("Baseline Cached Elemetns: ", base_node_data.sum())



            matching = sample_node_metadata & distinct_hit
            print("Distinct Cached Elemetns: ", matching.sum().item())
            print("Distinct Cache Percentage: ", matching.sum().item() / sample_node_metadata.sum())

            matching2 = base_node_data & distinct_hit
            print("Distinct Base Elemetns: ", matching2.sum().item())
            print("Distinct Base Cache Percentage: ", matching2.sum().item() / base_node_data.sum())
            break
            # print("Baseline", base_node_data)
            # print("Cached", sample_node_metadata)


            


    #     if(step == 100):
    #         sorted_items = sorted(counts_dict.items(), key=lambda item: item[1], reverse=True)

    #         # Convert the sorted items back into a dictionary if you want to preserve the order
    #         from collections import OrderedDict
    #         sorted_dict = OrderedDict(sorted_items)
    #         #print("sorted_dict: ", sorted_dict)

    #         count = 0
    #         ele_count = 0
    #         # for key, value in sorted_dict.items():
    #         #     print(f"{key}: {value}")
    #         #     ele_count += value
    #         #     count += 1
    #         #     if count >= 100:
    #         #         break

    #         # print("ele count: ", ele_count)
    #         # total_counts = sum(sorted_dict.values())
    #         # print("total count: ", total_counts)

    #         break

    #   #  print("src nodes len: ", len(blocks[0].srcdata[dgl.NID]))
    #   #  print("blocks: ", blocks)




   # seed_nodes_num = 1
   # fanout = 2
   # edge_dir = "in"
   # seed_nodes = np.random.randint(0, g.num_nodes(), seed_nodes_num)

  #    out = sample_neighbors2(
  #          g, seed_nodes, fanout, edge_dir=edge_dir
  #      )
  #  print("Sample test: ", out)
    



