"""
SynergyNet

chemical hypergraph
dose hypergraph
"""

import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import HyperGAT_Model as srcHGATmdl

class DrugCombHypergraph(nn.Module):
    def __init__(self, in_chemical_n,
                       in_dose_n,
                       in_onehot_n,
                       out_syn_n,
                       out_region_n,
                       n_hid,
                       dropout=0.5, 
                       fc_dropout=0.5, 
                       join=False, 
                       atype='additive',
                       device=None,
                       use_auxi=False):
        """construct model"""
        super(DrugCombHypergraph, self).__init__()
        self.in_chemical_n = in_chemical_n
        self.in_dose_n = in_dose_n
        self.in_onehot_n = in_onehot_n
        self.out_syn_n = out_syn_n
        self.out_region_n = out_region_n
        self.n_hid = n_hid
        self.dropout = dropout
        self.fc_dropout = fc_dropout
        self.join = join
        self.atype = atype
        self.device = device
        self.use_auxi = use_auxi

        # dose hypergraph layers
        self.dose_hgc = srcHGATmdl.HGAT_subg(in_ch_n=self.in_dose_n,
                                             n_hid=self.n_hid,
                                             dropout=self.dropout,
                                             device=self.device)
        # chemical hypergraph layers
        self.chemical_hgc = srcHGATmdl.HGAT_subg(in_ch_n=self.in_chemical_n,
                                                 n_hid=self.n_hid,
                                                 dropout=self.dropout,
                                                 device=self.device)




        # dropout layer for fully-connected layers
        self.fc_dropout = nn.Dropout(self.fc_dropout)
        
        # hidden layer
        if self.join:
            sg_hid = self.n_hid * 2
        else:
            sg_hid = self.n_hid

        # fully connected layers (concatenation)
        # task 1: synergy prediction
        # task 2: region prediction
        if self.in_onehot_n is not None:
            sg_hid = 2 * sg_hid + in_onehot_n
        else:
            sg_hid = 2 * sg_hid # i.e., two modules

        if self.use_auxi == True:
            # auxiliary task
            hid_fc1_aux = 2 * sg_hid // 3
            hid_fc2_aux = 2 * hid_fc1_aux // 3
            self.bn_aux = nn.BatchNorm1d(sg_hid)
            self.fc1_aux = nn.Linear(sg_hid, hid_fc1_aux)
            self.fc2_aux = nn.Linear(hid_fc1_aux, hid_fc2_aux)
            self.fc3_aux = nn.Linear(hid_fc2_aux, self.out_region_n)

            # main task
            sg_hid = sg_hid + hid_fc2_aux
            hid_fc1 = 2 * sg_hid // 3
            hid_fc2 = 2 * hid_fc1 // 3
            self.bn = nn.BatchNorm1d(sg_hid)
            self.task1_fc1 = nn.Linear(sg_hid, hid_fc1)
            self.task1_fc2 = nn.Linear(hid_fc1, hid_fc2)
            self.task1_fc3 = nn.Linear(hid_fc2, self.out_syn_n)
        else:
            # main task
            sg_hid = sg_hid
            hid_fc1 = 2 * sg_hid // 3
            hid_fc2 = 2 * hid_fc1 // 3
            self.bn1 = nn.BatchNorm1d(sg_hid)
            self.task1_fc1 = nn.Linear(sg_hid, hid_fc1)
            self.task1_fc2 = nn.Linear(hid_fc1, hid_fc2)
            self.task1_fc3 = nn.Linear(hid_fc2, self.out_syn_n)

    def forward(self, data_list):
        """make prediction"""
        # load sample
        chemical_list, dose_list, onehot_list = data_list

        # recode node loss and embeddings
        node_loss = 0
        emb_dict = {'chemical':None, 'dose':None}

        # forward pass
        # pass dose hypergraph
        x, xe, pair, G, sgs = dose_list
        dose_xsg, dose_node_loss, dose_attn  = self.dose_hgc(x, xe, pair, G, sgs)

        # pass chemical hypergraph
        x, xe, pair, G, sgs = chemical_list
        chemical_xsg, chemical_node_loss, chemical_attn = self.chemical_hgc(x, xe, pair, G, sgs)

        # concatenation
        if len(onehot_list) > 0:
            x = torch.concat( (chemical_xsg, dose_xsg, onehot_list[0]),1 )
        else:
            x = torch.concat( (chemical_xsg, dose_xsg),1 )
            
        # node loss
        loss = chemical_node_loss + dose_node_loss
        node_loss+=loss

        # subgraph embedding
        emb_dict['dose'] = (dose_xsg, dose_attn)
        emb_dict['chemical'] = (chemical_xsg, chemical_attn)
    
        # fully connected layers (concatenation)
        # task 1: synergy prediction
        # task 2: region prediction
        if self.use_auxi == True:
            # auxiliary task
            x_aux = self.bn_aux(x)
            x_aux = self.fc_dropout( F.elu(self.fc1_aux(x_aux)) )
            x_aux = self.fc_dropout( F.elu(self.fc2_aux(x_aux)) )
            x2 = self.fc3_aux(x_aux)

            # main task
            x1 = torch.concat( (x, x_aux),1 )
            x1 = self.bn(x1)
            x1 = self.fc_dropout( F.elu(self.task1_fc1(x1)) )
            x1 = self.fc_dropout( F.elu(self.task1_fc2(x1)) )
            x1 = self.task1_fc3(x1)
        else:
            # main task
            x1 = self.bn1(x)
            x1 = self.fc_dropout( F.elu(self.task1_fc1(x1)) )
            x1 = self.fc_dropout( F.elu(self.task1_fc2(x1)) )
            x1 = self.task1_fc3(x1)
            x2 = None

        return x1, x2, node_loss, emb_dict
