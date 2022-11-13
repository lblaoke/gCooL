import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.nn import GCNConv,SGConv,SAGEConv,GATConv,GraphConv,GINConv
from torch_geometric.utils import degree,to_networkx
from torch_scatter import scatter
import networkx as nx
import numpy as np
import random

def setup_seed(seed:int=None):
    torch.backends.cudnn.enabled = True
    if seed:
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        print('Random seed set to be: %d' % seed)
    else:
        torch.backends.cudnn.benchmark = True

def get_base_model(name:str):
    def gat_wrapper(in_channels,out_channels):
        return GATConv(
            in_channels     = in_channels           ,
            out_channels    = out_channels//4       ,
            heads           = 4
        )
    def gin_wrapper(in_channels,out_channels):
        mlp = nn.Sequential(
            nn.Linear(in_channels,2*out_channels)   ,
            nn.ELU()                                ,
            nn.Linear(2*out_channels,out_channels)
        )
        return GINConv(mlp)
    base_models = {
        'GCNConv'   : GCNConv                       ,
        'SGConv'    : SGConv                        ,
        'SAGEConv'  : SAGEConv                      ,
        'GATConv'   : gat_wrapper                   ,
        'GraphConv' : GraphConv                     ,
        'GINConv'   : gin_wrapper
    }

    return base_models[name]

def get_activation(name:str):
    activations = {
        'relu'          : F.relu                    ,
        'elu'           : F.elu                     ,
        'prelu'         : torch.nn.PReLU()          ,
        'rrelu'         : F.rrelu                   ,
        'selu'          : F.selu                    ,
        'celu'          : F.celu                    ,
        'leaky_relu'    : F.leaky_relu              ,
        'rrelu'         : torch.nn.RReLU()          ,
        'gelu'          : torch.nn.GELU()           ,
        'softplus'      : F.softplus                ,
        'tanh'          : F.tanh                    ,
        'sigmoid'       : F.sigmoid
    }

    return activations[name]

def compute_pr(edge_index,damp:float=0.85,k:int=10):
    num_nodes = edge_index.max().item() + 1
    deg_out = degree(edge_index[0])
    x = torch.ones((num_nodes,)).to(edge_index.device).to(torch.float32)

    for i in range(k):
        edge_msg = x[edge_index[0]]/deg_out[edge_index[0]]
        agg_msg = scatter(edge_msg, edge_index[1], reduce='sum')
        x = (1-damp)*x+damp*agg_msg

    return x

def eigenvector_centrality(data):
    graph = to_networkx(data)
    x = nx.eigenvector_centrality_numpy(graph)
    x = [x[i] for i in range(data.num_nodes)]
    return torch.tensor(x, dtype=torch.float32).to(data.edge_index.device)

def generate_split(num_samples:int,train_ratio:float,val_ratio:float):
    train_len = int(num_samples * train_ratio)
    val_len = int(num_samples * val_ratio)
    test_len = num_samples - train_len - val_len

    train_set, test_set, val_set = random_split(torch.arange(0, num_samples), (train_len, test_len, val_len))

    idx_train, idx_test, idx_val = train_set.indices, test_set.indices, val_set.indices
    train_mask = torch.zeros((num_samples,)).to(torch.bool)
    test_mask = torch.zeros((num_samples,)).to(torch.bool)
    val_mask = torch.zeros((num_samples,)).to(torch.bool)

    train_mask[idx_train] = True
    test_mask[idx_test] = True
    val_mask[idx_val] = True

    return train_mask, test_mask, val_mask

