import argparse, datetime
import dgl
import sklearn.metrics
import torch, torch.nn as nn, torch.optim as optim
import time, tqdm, numpy as np

from models import *
from dataloader import IGB260MDGLDataset
from Dist_GIDS_Loader import Simulation_Loader
from Graph_Coloring import Graph_Coloring

def color_graph(g, args, device):
    dim = args.emb_size

    g.ndata['features'] = g.ndata['feat']
    g.ndata['labels'] = g.ndata['label']
    
    
    Grah_Coloring_Tool = Graph_Coloring.Graph_Coloring()

    adj_csc = g.adj_tensors('csc')
    indptr = adj_csc[0]
    indices = adj_csc[1]


    num_nodes = g.number_of_nodes()
    color_tensor = torch.empty(num_nodes, dtype=torch.int64).contiguous()

    Grah_Coloring_Tool.set_color_buffer(color_tensor.data_ptr())
    Grah_Coloring_Tool.cpu_color_graph()
    
    num_colors = Grah_Coloring_Tool.get_num_color()

    topk_color_tensor =  torch.empty(num_colors * args.topk, dtype=torch.int64).contiguous()
    
    Grah_Coloring_Tool.set_topk_color_buffer(topk_color_tensor.data_ptr())
    Grah_Coloring_Tool.cpu_count_nearest_color()

    torch.save(topk_color_tensor, args.out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Loading dataset
    parser.add_argument('--path', type=str, default='/mnt/nvme14/IGB260M', 
        help='path containing the datasets')
    parser.add_argument('--dataset_size', type=str, default='tiny',
        choices=['tiny', 'small', 'medium', 'large', 'full'], 
        help='size of the datasets')
    parser.add_argument('--num_classes', type=int, default=19, 
        choices=[19, 2983, 172], help='number of classes')
    parser.add_argument('--in_memory', type=int, default=0, 
        choices=[0, 1], help='0:read only mmap_mode=r, 1:load into memory')
    parser.add_argument('--synthetic', type=int, default=0,
        choices=[0, 1], help='0:nlp-node embeddings, 1:random')
    parser.add_argument('--data', type=str, default='IGB')
    parser.add_argument('--emb_size', type=int, default=1024)

    #Output file format
    parser.add_argument('--out_path', type=str, default='./out.pt', 
        help='path for the output file')

    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--uva_graph', type=int, default=0)

    args = parser.parse_args()


    labels = None
    device = f'cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu'
    if(args.data == 'IGB'):
        print("Dataset: IGB")
        dataset = IGB260MDGLDataset(args)
        g = dataset[0]
        g  = g.formats('csc')
    elif(args.data == "OGB"):
        print("Dataset: OGB")
        dataset = OGBDGLDataset(args)
        g = dataset[0]
        g  = g.formats('csc')
    else:
        g=None
        dataset=None
    

    color_graph(g,args,device)
