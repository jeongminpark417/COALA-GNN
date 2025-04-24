import argparse, datetime
import dgl
#import sklearn.metrics
import torch, torch.nn as nn, torch.optim as optim
import time, tqdm, numpy as np

from models import *
from dataloader import IGB260MDGLDataset
from COALA_GNN import Graph_Coloring
from dataloader import load_ogb_graph

def color_graph(g, args, device):
    dim = args.emb_size

    # g.ndata['features'] = g.ndata['feat']
    # g.ndata['labels'] = g.ndata['label']
    
    
    num_nodes = g.number_of_nodes()
    print("number of nodes: ", num_nodes)
    Grah_Coloring_Tool = Graph_Coloring(num_nodes)

    adj_csc = g.adj_tensors('csc')
    indptr = adj_csc[0]
    indices = adj_csc[1]

    Grah_Coloring_Tool.set_adj_csc(indptr.data_ptr(), indices.data_ptr())
    print(f"indptr len: {len(indptr)} indices len: {len(indices)}")

    max_int64 = (1 << 63) - 1

    color_tensor = torch.zeros(num_nodes, dtype=torch.int64).contiguous()
    
    Grah_Coloring_Tool.set_color_buffer(color_tensor.data_ptr())
    #Grah_Coloring_Tool.cpu_color_graph()
    train_nid = torch.nonzero(g.ndata['train_mask'], as_tuple=True)[0]
    print("train nid: ", train_nid)
    Grah_Coloring_Tool.cpu_color_graph_optimized(train_nid.data_ptr(), len(train_nid))
    
    num_colors = Grah_Coloring_Tool.get_num_color()
    print(f"num_colors: {num_colors}")

    num_colored_nodes = Grah_Coloring_Tool.get_num_color_node()
    print(f"num colored node: {num_colored_nodes}")

    topk_color_tensor =  torch.zeros(num_colors * args.topk, dtype=torch.int64).contiguous()
    topk_affinity_tensor =  torch.zeros(num_colors * args.topk, dtype=torch.float64).contiguous()

    Grah_Coloring_Tool.set_topk_color_buffer(topk_color_tensor.data_ptr())
    Grah_Coloring_Tool.set_topk_affinity_buffer(topk_affinity_tensor.data_ptr())


    print("count nearest colors with less memory")
    #Grah_Coloring_Tool.cpu_count_nearest_color_less_memory()
    Grah_Coloring_Tool.cpu_calculate_color_affinity()

    del Grah_Coloring_Tool
    print("saving Numpy Arrays")
    np.save(args.out_path_color, color_tensor.cpu().numpy())
    topk_color_tensor = topk_color_tensor.reshape(num_colors, args.topk)
    np.save(args.out_path_topk, topk_color_tensor.cpu().numpy())
    topk_affinity_tensor = topk_affinity_tensor.reshape(num_colors, args.topk)
    np.save(args.out_path_affinity, topk_affinity_tensor.cpu().numpy())
    print("Saving is done")
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Loading dataset
    parser.add_argument('--path', type=str, default='/mnt/nvme14/IGB260M', 
        help='path containing the datasets')
    parser.add_argument('--dataset_size', type=str, default='experimental',
        choices=['experimental', 'small', 'medium', 'large', 'full'], 
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
    parser.add_argument('--out_path_color', type=str, default='./color.npy',
        help='path for the output color file')
    parser.add_argument('--out_path_topk', type=str, default='./topk.npy', 
        help='path for the output topk file')
    parser.add_argument('--out_path_affinity', type=str, default='./score.npy', 
        help='path for the output topk file')

    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--uva_graph', type=int, default=0)

    args = parser.parse_args()

    print("start")

    labels = None
    device = f'cuda:' + str(args.device) if torch.cuda.is_available() else 'cpu'
    if(args.data == 'IGB'):
        print("Dataset: IGB")
        dataset = IGB260MDGLDataset(args)
        g = dataset[0]
        g  = g.formats('csc')
    elif(args.data == "OGB"):
        print("Dataset: OGB")
        g, labels = load_ogb_graph("ogbn-papers100M", args.path)

    else:
        g=None
        dataset=None
    
    print("Setup done, start graph coloring")
    color_graph(g,args,device)

