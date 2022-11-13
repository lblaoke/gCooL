import argparse
import os.path as osp
import nni
import torch
from torch_geometric.utils import degree,to_undirected
from sp import SimpleParam
from src.model import *
from src.functional import *
from src.eval import *
from src.utils import *
from src.dataset import get_dataset
from time import time

# randomly drop edges by weights
def drop_edge(idx):
    global drop_weights

    assert param['drop_scheme'] in ['degree','evc','pr'], 'Unimplemented drop scheme!'
    return drop_edge_weighted(data.edge_index,drop_weights,p=param[f'drop_edge_rate_{idx}'],threshold=0.7)

# training per epoch
def train_epoch(epoch):
    net.train()

    # drop edges
    edge_index_1,edge_index_2 = drop_edge(1),drop_edge(2)

    # drop features
    x1 = drop_feature_weighted_2(data.x,feature_weights,param['drop_feature_rate_1'])
    x2 = drop_feature_weighted_2(data.x,feature_weights,param['drop_feature_rate_2'])

    # contrastive training
    loss = net.fit(epoch,opt,x1,x2,edge_index_1,edge_index_2)
    return loss.item()

# testing
def test(device):
    net.eval()
    global split

    # encoding
    with torch.no_grad():
        z = net(data.x,data.edge_index)

    # classifier on generated features
    if args.dataset=='WikiCS':
        micro_f1,macro_f1 = [],[]
        for i in range(20):
            result = log_regression(z,data.y,dataset,1000,device,split=f'wikics:{i}')
            micro_f1.append(result["Micro-F1"])
            macro_f1.append(result["Macro-F1"])
        micro_f1 = sum(micro_f1)/len(micro_f1)
        macro_f1 = sum(macro_f1)/len(macro_f1)
    else:
        result = log_regression(z,data.y,dataset,5000,device,split='rand:0.1',preload_split=split)
        micro_f1 = result["Micro-F1"]
        macro_f1 = result["Macro-F1"]

    return micro_f1,macro_f1

if __name__ == '__main__':
    # hyper-parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='WikiCS')
    parser.add_argument('--param', type=str, default='local:wikics.json')
    default_param = {
        'learning_rate'         : 0.01      ,
        'num_hidden'            : 384       ,
        'num_proj_hidden'       : 384       ,
        'activation'            : 'prelu'   ,
        'base_model'            : 'GCNConv' ,
        'drop_edge_rate_1'      : 0.2       ,
        'drop_edge_rate_2'      : 0.2       ,
        'drop_feature_rate_1'   : 0.1       ,
        'drop_feature_rate_2'   : 0.1       ,
        'tau'                   : 0.4       ,
        'num_epochs'            : 1000      ,
        'weight_decay'          : 1e-5
    }
    param_keys = default_param.keys()
    for key in param_keys:
        parser.add_argument(f'--{key}',type=type(default_param[key]),nargs='?')
    args = parser.parse_args()

    # parse param
    sp = SimpleParam(default=default_param)
    param = sp(source=args.param,preprocess='nni')

    # merge cli arguments and parsed parameters
    for key in param_keys:
        if getattr(args,key):
            param[key] = getattr(args,key)

    # set random seed and computing device
    setup_seed(39789)
    device = torch.device(args.device)

    # load dataset
    path = osp.expanduser('~/datasets')
    path = osp.join(path, args.dataset)
    dataset = get_dataset(path,args.dataset)
    data = dataset[0].to(device)

    # generate split
    split = generate_split(data.num_nodes,train_ratio=0.1,val_ratio=0.1)

    # initiate models
    encoder = Encoder(
        dataset.num_features    ,
        param['hidden_layer']   ,
        param['num_hidden']     ,
        get_activation(param['activation'])
    ).to(device)
    net = gCooL(
        encoder         = encoder                   ,
        num_hidden      = param['num_hidden']       ,
        num_proj_hidden = param['num_proj_hidden']  ,
        tau             = param['tau']              ,
        num_community   = param['num_community']    ,
        lamda           = param['lamda']            ,
        gamma           = param['gamma']            ,
        stride          = param['stride']
    ).to(device)
    opt = torch.optim.Adam(net.parameters(),lr=param['learning_rate'],weight_decay=param['weight_decay'])

    # weights for dropping edges
    if param['drop_scheme'] == 'degree':
        drop_weights = degree_drop_weights(data.edge_index).to(device)
    elif param['drop_scheme'] == 'pr':
        drop_weights = pr_drop_weights(data.edge_index, aggr='sink', k=200).to(device)
    elif param['drop_scheme'] == 'evc':
        drop_weights = evc_drop_weights(data).to(device)
    else:
        drop_weights = None

    # weights for dropping features
    if param['drop_scheme'] == 'degree':
        edge_index_ = to_undirected(data.edge_index)
        node_deg = degree(edge_index_[1])
        if args.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_deg).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_deg).to(device)
    elif param['drop_scheme'] == 'pr':
        node_pr = compute_pr(data.edge_index)
        if args.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_pr).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_pr).to(device)
    elif param['drop_scheme'] == 'evc':
        node_evc = eigenvector_centrality(data)
        if args.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_evc).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_evc).to(device)
    else:
        feature_weights = torch.ones((data.x.size(1),)).to(device)

    # training and testing
    start = time()

    for epoch in range(param['num_epochs']+1):
        loss = train_epoch(epoch)
        if epoch%10==0:
            print(f'Training epoch: {epoch:04d}, loss = {loss:.4f}')

        if epoch%100==0:
            print()
            print(f'================ Test epoch: {epoch:04d} ================')
            micro_f1,macro_f1 = test(device)
            print(f'Micro-F1 = {micro_f1}')
            print(f'Macro-F1 = {macro_f1}')
            print()

    end = time()

    # time usage
    print(f'Finished in {(end-start)/60:.1f} min')
