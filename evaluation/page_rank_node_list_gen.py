import argparse, datetime
import dgl
import sklearn.metrics
import torch, torch.nn as nn, torch.optim as optim
import time, tqdm, numpy as np
from models import *
from dataloader import IGB260MDGLDataset, OGBDGLDataset
import csv 
import warnings

import torch.cuda.nvtx as t_nvtx
import nvtx
import threading
import gc

import GIDS

from ogb.graphproppred import DglGraphPropPredDataset
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

import networkx as nx
import dgl.function as fn

torch.manual_seed(0)
dgl.seed(0)
warnings.filterwarnings("ignore")

    
def compute_pagerank(g, DAMP, K, N):
    g.ndata['pv'] = torch.ones(N) / N
    degrees = g.in_degrees(g.nodes()).type(torch.float32)
    for k in range(K):
        g.ndata['pv'] = g.ndata['pv'] / degrees
        g.update_all(fn.copy_u('pv','m'),
                     fn.sum('m', 'pv'))
        g.ndata['pv'] = (1 - DAMP) / N + DAMP * g.ndata['pv']
    return g.ndata['pv'] 

def normalize_pagerank_to_uint8(pv):
    N = len(pv)  # Number of nodes
    # Sort PageRank values and get indices
    sorted_indices = torch.argsort(pv, descending=False)
    # Create an empty tensor for rank-based normalized values
    normalized_ranks = torch.zeros_like(pv)
    
    # Assign values based on rank
    for rank, idx in enumerate(sorted_indices):
        # Normalize rank to 0-255 range; note that we use rank/(N-1) to ensure the last element gets a value close to 0.
        normalized_value = 255 * rank / (N - 1)
        normalized_ranks[idx] = normalized_value

    # Convert to uint8
    #normalized_ranks_uint8 = normalized_ranks.to(torch.uint8)
    normalized_ranks_uint8 = normalized_ranks.numpy().astype(np.uint8)

    return normalized_ranks_uint8

def normalize_pagerank_to_uint16(pv):
    N = len(pv)  # Number of nodes
    # Sort PageRank values and get indices
    sorted_indices = torch.argsort(pv, descending=False)
    # Create an empty tensor for rank-based normalized values
    normalized_ranks = torch.zeros_like(pv)
    
    # Assign values based on rank
    for rank, idx in enumerate(sorted_indices):
        # Normalize rank to 0-255 range; note that we use rank/(N-1) to ensure the last element gets a value close to 0.
        normalized_value = 65535 * rank / (N - 1)
        normalized_ranks[idx] = normalized_value

    normalized_ranks_uint16 = normalized_ranks.numpy().astype(np.uint16)
    return normalized_ranks_uint16

def save_to_binary_file(filepath, data_uint16):
    # Data is already a numpy array of type uint16
    with open(filepath, 'wb') as file:
        file.write(data_uint16.tobytes())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Loading dataset
    parser.add_argument('--path', type=str, default='/mnt/nvme14/IGB260M', 
        help='path containing the datasets')
    parser.add_argument('--dataset_size', type=str, default='experimental',
        choices=['experimental', 'small', 'medium', 'large', 'full'], 
        help='size of the datasets')
    parser.add_argument('--num_classes', type=int, default=19, 
        choices=[19, 2983, 171,172, 173], help='number of classes')
    parser.add_argument('--in_memory', type=int, default=0, 
        choices=[0, 1], help='0:read only mmap_mode=r, 1:load into memory')
    parser.add_argument('--synthetic', type=int, default=0,
        choices=[0, 1], help='0:nlp-node embeddings, 1:random')

    parser.add_argument('--out_path', type=str, default='./pr_full.pt', 
    help='output path for the node list')

    parser.add_argument('--damp', type=float, default=0.85, 
    help='Damp value for the page rank algorithm')
    parser.add_argument('--K', type=int, default=20, 
    help='K value for the page rank algorithm')
    parser.add_argument('--data', type=str, default='IGB')
    parser.add_argument('--uva_graph', type=int, default=0)
    parser.add_argument('--emb_size', type=int, default=1024)

   

    args = parser.parse_args()
    
    labels = None
   # device = f'cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu'
    if(args.data == 'IGB'):
        print("Dataset: IGB")
        dataset = IGB260MDGLDataset(args)
        g = dataset[0]
        g  = g.formats('csc')
    elif(args.data == "OGB"):
        print("Dataset: OGB")
        dataset = OGBDGLDataset(args)
        g = dataset[0]
    else:
        g=None
        dataset=None

    N = g.number_of_nodes()

    pr_val = compute_pagerank(g, args.damp, args.K, N)
    norm_pv = normalize_pagerank_to_uint8(pr_val)
    #norm_pv = normalize_pagerank_to_uint16(pr_val)
    save_to_binary_file(args.out_path, norm_pv)

    #print("pv: ",pr_val)
    #norm_pv = normalize_pagerank_to_uint8(pr_val)

    # topk = int(N * 0.6)
    # _, indices = torch.topk(pv, k=topk, largest=True)
    #print(norm_pv)
    #torch.save(pv, args.out_path)

