import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from math import exp
from src.functional import cos_sim,RBF_sim

class Encoder(nn.Module):
    def __init__(self,in_channels,hidden_layer,out_channels,activation):
        super(Encoder,self).__init__()
        self.activation = activation
        self.conv = nn.ModuleList([
            GCNConv(in_channels,hidden_layer),
            GCNConv(hidden_layer,out_channels)
        ])
    def forward(self,x,edge_index):
        for layer in self.conv:
            x = self.activation(layer(x,edge_index))
        return x

class Projection(nn.Sequential):
    def __init__(self,num_hidden,num_proj_hidden):
        super(Projection,self).__init__(
            nn.Linear(num_hidden,num_proj_hidden),
            nn.ELU(),
            nn.Linear(num_proj_hidden,num_hidden)
        )
    def forward(self,x):
        x = super(Projection,self).forward(x)
        return F.normalize(x)

class gCooL(nn.Module):
    def __init__(
            self                            ,
            encoder         : nn.Module     ,
            num_hidden      : int           ,
            num_proj_hidden : int           ,
            tau             : float = 0.4   ,
            num_community   : int   = 10    ,
            lamda           : float = 1.0   ,
            gamma           : float = 8e-5  ,
            alpha_rate      : float = 0.2   ,
            stride          : int   = 500   ,
            similarity      : str   = 'cos'
        ):
        super(gCooL,self).__init__()

        # hyper-parameter
        self.tau = tau
        self.num_community = num_community
        self.lamda = lamda
        self.gamma = gamma
        self.alpha_rate = alpha_rate
        self.stride = stride

        # similarity measure
        assert similarity in {'cos','RBF'}, 'Unknown similarity measure!'
        self.similarity = similarity

        # backbones
        self.encoder = encoder
        self.center = nn.Parameter(torch.randn(self.num_community,num_hidden,dtype=torch.float32))
        self.proj = Projection(num_hidden,num_proj_hidden)

    def forward(self,x,edge_index):
        return self.encoder(x,edge_index)

    def community_assign(self,h):
        if self.similarity=='cos':
            return cos_sim(h.detach(),F.normalize(self.center),self.tau,norm=True)
        return RBF_sim(h.detach(),F.normalize(self.center),self.tau,norm=True)

    def node_contrast(self,h1,h2):
        # compute similarity
        s12 = cos_sim(h1,h2,self.tau,norm=False)
        s21 = s12

        # compute InfoNCE
        loss12 = -torch.log(s12.diag())+torch.log(s12.sum(1))
        loss21 = -torch.log(s21.diag())+torch.log(s21.sum(1))
        L_node = (loss12+loss21)/2

        return L_node.mean()

    def DeCA(self,R,edge_index):
        n = len(R)
        m = n*(n-1)

        # adjacent matrix
        A = torch.zeros(n,n,device=R.device,dtype=torch.float32)
        A[edge_index[0],edge_index[1]] = 1

        # edge density constraint
        DF = R.T@A@R
        return (self.lamda*DF.sum()-(n-1+self.lamda)*DF.trace())/m+(R.T@R).trace()/n/2

    def community_contrast(self,h1,h2,R1,R2):
        # gather communities
        index1,index2 = R1.detach().argmax(dim=1),R2.detach().argmax(dim=1)
        C1,C2 = [],[]
        for i in range(self.num_community):
            h_c1,h_c2 = h1[index1==i],h2[index2==i]
            if h_c1.shape[0]>0:
                C1.append(h_c1.sum(dim=0)/h_c1.shape[0])
            else:
                C1.append(torch.zeros(h1.shape[1],device=h1.device,dtype=torch.float32))
            if h_c2.shape[0]>0:
                C2.append(h_c2.sum(dim=0)/h_c2.shape[0])
            else:
                C2.append(torch.zeros(h2.shape[1],device=h2.device,dtype=torch.float32))
        C1,C2 = torch.stack(C1),torch.stack(C2)

        # compute similarity
        if self.similarity=='cos':
            s_h1_c2 = cos_sim(h1,C2.detach(),self.tau,norm=False)
            s_h2_c1 = cos_sim(h2,C1.detach(),self.tau,norm=False)
            ws_h1_c2 = s_h1_c2
            ws_h2_c1 = s_h2_c1
        else:
            s_h1_c2 = RBF_sim(h1,C2.detach(),self.tau,norm=False)
            s_h2_c1 = RBF_sim(h2,C1.detach(),self.tau,norm=False)
            h1_extend,h2_extend = torch.stack([C1.detach()]*len(C1)),torch.stack([C2.detach()]*len(C2))
            h1_sub,h2_sub = h1_extend-h1_extend.transpose(0,1),h2_extend-h2_extend.transpose(0,1)
            w1,w2 = torch.exp(-self.gamma*(h1_sub*h1_sub).sum(2)),torch.exp(-self.gamma*(h2_sub*h2_sub).sum(2))
            ws_h1_c2 = s_h1_c2*w2[index2]
            ws_h2_c1 = s_h2_c1*w1[index1]

        # node-community contrast
        self_s12 = s_h1_c2.gather(1,index2.unsqueeze(-1)).squeeze(-1)
        self_s21 = s_h2_c1.gather(1,index1.unsqueeze(-1)).squeeze(-1)
        loss12 = -torch.log(self_s12)+torch.log( \
            self_s12+ \
            ws_h1_c2.sum(1)-ws_h1_c2.gather(1,index2.unsqueeze(-1)).squeeze(-1)
        )
        loss21 = -torch.log(self_s21)+torch.log( \
            self_s21+ \
            ws_h2_c1.sum(1)-ws_h2_c1.gather(1,index1.unsqueeze(-1)).squeeze(-1)
        )
        L_community = (loss12+loss21)/2

        return L_community.mean()

    def fit(self,epoch,opt,x1,x2,edge_index_1,edge_index_2):
        opt.zero_grad()

        # node contrast
        z1,z2 = self(x1,edge_index_1),self(x2,edge_index_2)
        h1,h2 = self.proj(z1),self.proj(z2)
        L_node = self.node_contrast(h1,h2)

        # community contrast
        R1,R2 = self.community_assign(h1),self.community_assign(h2)
        DeCA = (self.DeCA(R1,edge_index_1)+self.DeCA(R2,edge_index_2))/2
        L_community = self.community_contrast(h1,h2,R1,R2)

        # joint objective
        alpha = max(0,1-epoch/self.alpha_rate/self.stride)
        coef = exp(-epoch/self.stride)
        loss = L_node+alpha*(coef*DeCA+(1-coef)*L_community)

        loss.backward()
        opt.step()

        return loss

class LogReg(nn.Module):
    def __init__(self,in_channel:int,num_class:int):
        super(LogReg,self).__init__()
        self.fc = nn.Linear(in_channel,num_class)
        for m in self.modules():
            self.weights_init(m)
    def weights_init(self,m):
        if isinstance(m,nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
    def forward(self,seq):
        return self.fc(seq)
