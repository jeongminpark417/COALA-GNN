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

import GIDS
from GIDS import ID_Loader

from GIDS import GIDS_NVShmem_Loader 

#from custom_neighbor import sample_neighbors2

from dataloader import IGB260MDGLDataset, OGBDGLDataset, load_ogb, IGB260MDGLDataset_No_Feature

#from custom_sampler import NeighborSampler2

import ctypes

from mpi4py import MPI
import torch.distributed as dist

#import BAM_Feature_Store 



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

def track_acc_GIDS( g, args, label_array=None):

    in_feats = args.cache_dim

    # GIDS_Loader = None

    GIDS_Loader = GIDS.NVSHMEM_GIDS(
        page_size = args.page_size,
        off = args.offset,
        num_ele = args.num_ele,
        num_ssd = args.num_ssd,
        GPU_cache_size = args.GPU_cache_size,
        CPU_cache_size = args.CPU_cache_size,
        cache_dim = args.cache_dim,
        dim = args.cache_dim,
        num_ways=32,
        fan_out = [int(fanout) for fanout in args.fan_out.split(',')],
        batch_size = args.batch_size,
        use_nvshmem_tensor = False,
        nvshmem_test = False
    )

    rank = GIDS_Loader.get_rank()
    size = GIDS_Loader.get_world_size()
    device = 'cuda:' + str(rank)
    print("Rank: {rank} device: {device} world size:{size}")


    dist.init_process_group("nccl", world_size=size, rank=rank)
    

    if args.model_type == 'gcn':
        model = GCN(in_feats, args.hidden_channels, args.num_classes, 
            args.num_layers).to(device)
    if args.model_type == 'sage':
        #model = SAGE(in_feats, args.hidden_channels, args.num_classes, 
        #    args.num_layers).to(device)
        model = DistSAGE(in_feats, args.hidden_channels, args.num_classes, args.num_layers, torch.nn.functional.relu
            ).to(device)
    if args.model_type == 'gat':
        model = GAT(in_feats, args.hidden_channels, args.num_classes, 
            args.num_layers,  args.num_heads).to(device)

    #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=device, find_unused_parameters=True)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=device)


    ssd_list = []
    for i in range(args.num_ssd):
        ssd_list.append(rank * args.num_ssd + i)

    GIDS_Loader.init_cache(device_id=rank, ssd_list=ssd_list)

    #g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']

    train_nid = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
    val_nid = torch.nonzero(g.ndata['val_mask'], as_tuple=True)[0]
    test_nid = torch.nonzero(g.ndata['test_mask'], as_tuple=True)[0]
    
    my_train_nid = train_nid[rank::size]
    print(f"train nid: {train_nid}")
    print(f"Rank: {rank} my train nid: {my_train_nid}")

    sampler = dgl.dataloading.MultiLayerNeighborSampler(
               [int(fanout) for fanout in args.fan_out.split(',')]
               )

    #my_train_nid = train_nid

    train_dataloader = GIDS_NVShmem_Loader(
        g,
        my_train_nid,
        sampler,
        args.batch_size,
        args.cache_dim,
        GIDS_Loader,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        use_alternate_streams=False
        #device=device
        )


    test_dataloader = GIDS_NVShmem_Loader(
        g,
        test_nid,
        sampler,
        args.batch_size,
        args.cache_dim,
        GIDS_Loader,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        use_alternate_streams=False
        #device=device
        )


    loss_fcn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.learning_rate, weight_decay=args.decay
        )

    warm_up_iter = 300


    dummy_tensor = torch.empty((1024 * 1024, 1024), dtype=torch.float32, device='cuda')
    del dummy_tensor  # Free the dummy tensor but keep the memory allocated
    torch.cuda.empty_cache()  # Optional: Clear unnecessary cached blocks


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
        epoch_time_start = time.time()

        num_step = 0
        for step, (input_nodes, seeds, blocks, ret) in enumerate(train_dataloader):
            #print(f"Rank; {rank} step: {step}")
      
            # cpu_ret = ret.to("cpu")
            # print(f"Rank: {rank} step: {step} cpu_ret: {cpu_ret} ret: {ret}")
            # if(step % 100 == 0):
            #     print("step: ", step)
            # if(rank == 1):
            #     print("input_nodes: ", input_nodes)
            #     ret_cpu = ret.to("cpu")
            #     for i in range(len(input_nodes)):
            #         correct = g.ndata['feat'][input_nodes[i]]
            #         out = ret_cpu[i]
            #         cur_idx = input_nodes[i]
            #         print(f"IDX: {cur_idx} correct: ", correct)
            #         print(f"IDX: {cur_idx} ret: ", out)

            # break
            
            # if(step == warm_up_iter):
            #     print("warp up done")
            #     train_dataloader.print_stats()
            #     train_dataloader.print_timer()
            #     batch_input_time = 0
            #     transfer_time = 0
            #     train_time = 0
            #     e2e_time = 0
            #     e2e_time_start = time.time()
        
            
            # Features are fetched by the baseline GIDS dataloader in ret

            batch_inputs = ret
            transfer_start = time.time() 

            batch_labels = blocks[-1].dstdata['labels']
            

            blocks = [block.int().to(device) for block in blocks]
            batch_labels = batch_labels.to(device)
            transfer_time = transfer_time +  time.time()  - transfer_start

            train_start = time.time()
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_end_tme = time.time()
            train_time_iter =  train_end_tme - train_start               
            train_time += train_time_iter
            num_step += 1

        
        print(f"Performance for epoch: %{epoch}")
        e2e_time += time.time() - e2e_time_start 
        train_dataloader.print_stats()
        train_dataloader.print_timer()
        #print_times(transfer_time, train_time, e2e_time)
        print(f"Rank: {rank} num_step: {num_step} train_time: {train_time} ")
        batch_input_time = 0
        transfer_time = 0
        train_time = 0
        e2e_time = 0
    

        epoch_time = time.time() - epoch_time_start
        print("epoch time: ", epoch_time)
        #train_dataloader.print_stats()

    model.eval()
    predictions = []
    labels = []

    c = 0
    with torch.no_grad():
        #for _, _, blocks in test_dataloader:
        for step, (input_nodes, seeds, blocks, ret) in enumerate(test_dataloader):
            if(c % 100 == 0):
                print("Eval step: ", c)
            blocks = [block.to(device) for block in blocks]
            #inputs = blocks[0].srcdata['feat']
            inputs = ret

            batch_labels = blocks[-1].dstdata['labels']
            

            labels.append(blocks[-1].dstdata['label'].cpu().numpy())
            predictions.append(model(blocks, inputs).argmax(1).cpu().numpy())
            c += 1
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
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--log_every', type=int, default=2)

    
    parser.add_argument('--page_size', type=int, default=8)
    parser.add_argument('--offset', type=int, default=0, help='Offset for the feature data stored in the SSD') 
    parser.add_argument('--num_ele', type=int, default=100, help='Number of elements in the dataset (Total Size / sizeof(Type)') 
    parser.add_argument('--cache_dim', type=int, default=1024) #CHECK
    parser.add_argument('--num_ssd', type=int, default=1) 
    parser.add_argument('--GPU_cache_size', type=int, default=4) 
    parser.add_argument('--CPU_cache_size', type=int, default=0) 



    parser.add_argument('--eviction_policy', type=int, default=0)




    args = parser.parse_args()
    labels = None
    device = 'cuda:' + str(args.device)


    in_feats = args.cache_dim


    # GIDS_Loader = GIDS.NVSHMEM_GIDS(
    #     page_size = args.page_size,
    #     off = args.offset,
    #     num_ele = args.num_ele,
    #     num_ssd = args.num_ssd,
    #     cache_size = args.cache_size,
    #     cache_dim = args.cache_dim,
    #     dim = args.cache_dim,
    #     num_ways=32,
    #     fan_out = [int(fanout) for fanout in args.fan_out.split(',')],
    #     batch_size = args.batch_size,
    #     use_nvshmem_tensor = True
    # )


    print("device: ", device)
    if(args.data == 'IGB'):
        print("Dataset: IGB")
        dataset = IGB260MDGLDataset_No_Feature(args)
        #dataset = IGB260MDGLDataset(args)

        g = dataset[0]

    elif(args.data == 'OGB'):
        print("Dataset: OGB")
        # dataset = OGBDGLDataset(args)
        # g = dataset[0]
        # g  = g.formats('csc')

        g, labels = load_ogb("ogbn-papers100M", args.path)
   # g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']

    

    print(f"RANK = {os.environ.get('RANK')}")
    print(f"WORLD_SIZE = {os.environ.get('WORLD_SIZE')}")
    print(f"MASTER_ADDR = {os.environ.get('MASTER_ADDR')}")
    print(f"MASTER_PORT = {os.environ.get('MASTER_PORT')}")

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'


    # rank = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    # size = int(os.environ['OMPI_COMM_WORLD_RANK'])

    # # Alternatively, use other environment variables based on MPI implementation
    # # rank = int(os.environ.get('PMI_RANK', -1))           # For MPICH
    # # rank = int(os.environ.get('MPI_LOCALRANKID', -1))    # For Intel MPI

    # print(f"Process {rank} of {size}")

    track_acc_GIDS( g, args, labels)

    # MPI.Finalize()

