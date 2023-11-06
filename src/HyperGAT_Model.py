"""
HyperGAT class
  HGAT_sparse
  HGNN_sg_attn
  HGNN_fc
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import numpy as np, scipy.sparse as sp
import cupy as cp
from torch.utils.dlpack import from_dlpack
import HyperGAT_Model as srcHGATmdl
import HGNN_Attention as attention

class HGAT_sparse(nn.Module):
    """
    adopted from: https://github.com/luoyuanlab/SHINE/blob/main/layers.py
    """

    def __init__(self, in_ch_n, out_ch, dropout, alpha, transfer, concat=True, bias=False, coarsen=False):
        super(HGAT_sparse, self).__init__()
        self.e_dropout = nn.Dropout(dropout)
        self.in_ch_n = in_ch_n
        self.out_ch = out_ch
        self.alpha = alpha
        self.concat = concat
        
        self.transfer = transfer

        if self.transfer:
            self.wt = Parameter(torch.Tensor(self.in_ch_n, self.out_ch))
        else:
            self.register_parameter('wt', None)
        

        if bias:
            self.bias = Parameter(torch.Tensor(1, self.out_ch))
        else:
            self.register_parameter('bias', None)        
        
        self.coarsen = coarsen

        self.reset_parameters()

    def reset_parameters(self): 
        stdv = 1. / math.sqrt(self.out_ch)
        if self.wt is not None:
            self.wt.data.uniform_(-stdv, stdv)

        
        
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    

    def reset_parameters_xavier(self): 
        if self.wt is not None:
            nn.init.xavier_uniform_(self.wt)

        
        if self.bias is not None:
            nn.init.xavier_uniform_(self.bias)         
        

    def std_scale(self, x):
        xstd = x.std(1, unbiased=False, keepdim=True)
        xstd = torch.where(xstd>0, xstd, torch.tensor(1., device=x.device)) 
        x = (x - x.mean(1, keepdim=True)) / xstd
        return x

    def forward(self, x, xe, pair, a, val=None, e_degs=None, n_degs=None): 
        
        
        if self.transfer:
            x = torch.mm(x, self.wt)
            xe = torch.mm(xe, self.wt)
            
            if self.bias is not None:
                
                x = x + self.bias
                xe = xe + self.bias

        n_edge = xe.shape[0] 
        n_node = x.shape[0] 
        
        if val is None:
            pair_h = xe[ pair[0] ] * x[ pair[1] ]
        else:
            pair_h = xe[ pair[0] ] * x[ pair[1] ] * val
            
        if e_degs is not None:
            pair_h /= e_degs[ pair[0] ].sqrt().unsqueeze(-1)
        if n_degs is not None:
            pair_h /= n_degs[ pair[1] ].sqrt().unsqueeze(-1)
        pair_e = torch.mm(pair_h, a).squeeze() 
        

        e = torch.zeros(n_edge, n_node, device=pair.device)
        e[pair[0], pair[1]] = torch.exp(pair_e)
        e = torch.log(1e-10 + self.e_dropout(e))
        
        
        attention_edge = F.softmax(e, dim=1) 

        xe_out = torch.mm(attention_edge, x)

        attention_node = F.softmax(e.transpose(0,1), dim=1)

        x = torch.mm(attention_node, xe) 


        if self.concat:
            x = F.elu(x)
            xe_out = F.elu(xe_out)
        else:
            x = F.relu(x)
            xe_out = F.relu(xe_out)

        
        if self.coarsen:
            return x, xe_out, torch.exp(e.T) 
        else:
            return x, xe_out

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_ch_n) + ' -> ' + str(self.out_ch) + ')' 


class HGNN_sg_attn(nn.Module):
    
    def __init__(self, vdim, mdim, atype='additive'):
        super(HGNN_sg_attn, self).__init__()
        self.attn_vector = torch.nn.Parameter(torch.zeros((vdim,1), dtype=torch.float), requires_grad=True)   
        
        stdv = 1. / math.sqrt(vdim) 
        self.attn_vector.data.uniform_(-stdv, stdv)


    def forward(self, x, sgs):
        xsize = list(x.size())
        bsize = sgs.shape[0]
        attn_wts = torch.matmul(x, self.attn_vector) 
        attn_wts = attn_wts.squeeze().unsqueeze(0).expand(bsize, xsize[0]) 
        x = torch.matmul(sgs*attn_wts, x)
        #print(f'attention weight: {attn_wts.shape} | subgraph: {sgs.shape} | x:{x.shape}')
        #print(attn_wts)
        #print(sgs)
        return x, attn_wts


class HGNN_sg_attn_multiplicative(nn.Module):

    def __init__(self, vdim, mdim, atype='multiplicative'):
        super(HGNN_sg_attn_multiplicative, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(mdim, vdim), requires_grad=True)
        nn.init.xavier_uniform_(self.W)


    def forward(self, x, sgs):
        x = sgs.matmul(x).matmul(self.W)
        return x, self.W

class HGNN_fc(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HGNN_fc, self).__init__()
        self.fc = nn.Linear(in_ch, out_ch)

    def forward(self, x):
        return self.fc(x)


class HGAT_subg(nn.Module):
    def __init__(self, in_ch_n, n_hid, dropout=0.5, fc_dropout=0.5, join=False, atype='additive', device=None):
        """construct model"""
        super(HGAT_subg, self).__init__()

        self.in_ch_n = in_ch_n
        self.n_hid = n_hid
        self.dropout = dropout
        self.fc_dropout = fc_dropout
        self.join = join
        self.atype = atype
        self.device = device
     
        self.hgc1 = srcHGATmdl.HGAT_sparse(self.in_ch_n, self.n_hid, dropout=self.dropout,
                                           alpha=0.2, transfer=True, bias=True, concat=False)
        self.hgc2 = srcHGATmdl.HGAT_sparse(self.n_hid, self.n_hid, dropout=self.dropout,
                                           alpha=0.2, transfer=True, bias=True, concat=False)

        if self.join:
            sg_hid = n_hid * 2
        else:
            sg_hid = n_hid
         
        if self.atype == 'additive':
            self.sga = srcHGATmdl.HGNN_sg_attn(sg_hid, sg_hid, self.atype)
        elif self.atype == 'multiplicative':
            self.sga = HGNN_sg_attn_multiplicative(sg_hid, sg_hid, self.atype)
        self.sga_dropout = nn.Dropout(self.dropout)

        l_hid = 2 * sg_hid // 3

        self.a1 = nn.Parameter( torch.zeros(size=(n_hid,1)) )
        self.a2 = nn.Parameter( torch.zeros(size=(n_hid,1)) )

        stdv = 1. / math.sqrt(n_hid)
        self.a1.data.uniform_(-stdv, stdv)
        self.a2.data.uniform_(-stdv, stdv)


    def forward(self, x, xe, pair, G, sgs=None):
        """make prediction"""
        
        # forward pass
        x1, xe = self.hgc1(x, xe, pair, self.a1)
        x, xe = self.hgc2(x1, xe, pair, self.a2)
        
        if self.join:
            x = torch.cat( (x, x1),1 )

        # get node loss
        x2 = x / torch.norm(x, p=2, dim=1, keepdim=True)
        xxt = torch.matmul(x2, x2.T)
        node_loss = ( (2 - 2*xxt)*G ).sum()
        
        # subg
        xsg, attn_wts = self.sga(x, sgs)
        xsg = self.sga_dropout(xsg)
        #print(f'x={x.shape} | xe={xe.shape} | xsg={xsg.shape} | attn={attn_wts.shape}')
        #print(x)
        #print(xe)
        #print(attn_wts)
        return xsg, node_loss, attn_wts
