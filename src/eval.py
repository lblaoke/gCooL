import torch
from torch.optim import Adam
import torch.nn.functional as F
from sklearn.metrics import f1_score
from src.model import LogReg

def get_idx_split(dataset,split,preload_split):
    if split[:4]=='rand':
        train_ratio = float(split.split(':')[1])
        num_nodes = dataset[0].x.size(0)
        train_size = int(num_nodes*train_ratio)
        indices = torch.randperm(num_nodes)
        return {
            'train' : indices[:train_size]              ,
            'val'   : indices[train_size:2*train_size]  ,
            'test'  : indices[2*train_size:]
        }
    elif split=='ogb':
        return dataset.get_idx_split()
    elif split.startswith('wikics'):
        split_idx = int(split.split(':')[1])
        return {
            'train' : dataset[0].train_mask[:,split_idx]    ,
            'test'  : dataset[0].test_mask                  ,
            'val'   : dataset[0].val_mask[:,split_idx]
        }
    elif split=='preloaded':
        assert preload_split, 'preloaded_split not found!'
        train_mask,test_mask,val_mask = preload_split
        return {
            'train' : train_mask    ,
            'test'  : test_mask     ,
            'val'   : val_mask
        }

    return None

def f1(y_true,y_pred,avg:str):
    y_true = y_true.view(-1)
    y_pred = y_pred.argmax(-1)
    return f1_score(y_true.cpu(),y_pred.cpu(),average=avg)

def evaluate(res):
    return {
        "Micro-F1":f1(**res,avg="micro"),
        "Macro-F1":f1(**res,avg="macro")
    }

def log_regression(z,y,dataset,num_epochs:int,device,split="rand:0.1",preload_split=None):
    # classifier settings
    num_hidden = z.size(1)
    num_classes = y.max().item()+1
    y = y.view(-1)
    classifier = LogReg(num_hidden,num_classes).to(device)
    optimizer = Adam(classifier.parameters(),lr=0.01,weight_decay=1e-5)

    # prepare splits
    split = get_idx_split(dataset,split,preload_split)
    split = {k: v.to(device) for k, v in split.items()}

    # obtain test results
    best_micro_f1,best_macro_f1 = 0,0

    for epoch in range(num_epochs+1):
        classifier.train()

        optimizer.zero_grad()
        output = classifier(z[split['train']])
        loss = F.cross_entropy(output,y[split['train']])
        loss.backward()
        optimizer.step()

        # update test results
        if epoch%5==0:
            classifier.eval()
            result = evaluate({
                'y_true': y[split['test']].view(-1,1),
                'y_pred': classifier(z[split['test']])
            })
            best_micro_f1 = max(best_micro_f1,result["Micro-F1"])
            best_macro_f1 = max(best_macro_f1,result["Macro-F1"])

    return {"Micro-F1":best_micro_f1,"Macro-F1":best_macro_f1}
