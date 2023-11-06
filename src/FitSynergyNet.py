"""
FitSynergyNet

Optimize AUC for classification task
Optimize RMSE for regression task
"""
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import sklearn.metrics as skmts
import sklearn.feature_extraction as skfet
import scipy.stats as scistat
from numba import jit
import cupy as cp

import torch
import torcheval.metrics as tchmts

def get_x_xe_pair(H, device):
    """ return x, xe, pair"""

    H = H.to(device)
    HT = H.T.to(device)
    HTa = HT/HT.sum(1, keepdim=True).to(device)
    HTa[HTa != HTa] = 0.0 #<-- fill nan with zero
    HTa = HTa.to(device)
    x = torch.eye(H.shape[0]).to(device)
    xe = HTa.mm(x).to(device)
    pair = HT.nonzero(as_tuple=False).t().to(device)
    return x, xe, pair

class DrugCombHypergraph:
    def __init__(self, chemical_H, chemical_G,
                       dose_H, dose_G,
                       batch_size, 
                       optimizer,
                       scheduler,
                       loss_fn,
                       loss_fn2,
                       device, 
                       task,
                       nratio=1,
                       use_auxi=False,
                       use_onehot=False):
        
        self.chemical_H = chemical_H
        self.chemical_G = chemical_G
        self.dose_H = dose_H
        self.dose_G = dose_G
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.loss_fn2 = loss_fn2
        self.device = device
        self.task = task
        self.nratio = nratio
        self.chemical = []
        self.dose = []
        self.use_auxi = use_auxi
        self.use_onehot = use_onehot
        
        # prepare H, G
        x, xe, pair = get_x_xe_pair(torch.Tensor(self.dose_H).to(self.device), self.device)
        G = torch.Tensor(self.dose_G).to(self.device)
        self.dose = [x, xe, pair, G]

        x, xe, pair = get_x_xe_pair(torch.Tensor(self.chemical_H).to(self.device), self.device)
        G = torch.Tensor(self.chemical_G).to(self.device)
        self.chemical = [x, xe, pair, G]

    def train(self, model, train_data, valid_data):
        # training
        model.train()
        n_data = train_data.get_length()
        btz_list = train_data.generate_batch(batch_size=self.batch_size)
        print(f'training mode')
        print(f'total data={n_data} | batch_size={self.batch_size} | #batches={len(btz_list)}')
        train_loss = 0.0
        true_list, pred_list = [], []
        for i in tqdm( range(0, len(btz_list)) ):
            # zero the parameter gradients
            self.optimizer.zero_grad()

            # get batch data & set device
            sample_dict = train_data.get_slice(btz_list[i])
            n_subject = len(btz_list[i])

            batch_dict = {}
            for dt, sample in sample_dict.items():
                if dt == 'region':
                    batch_dict['region'] = torch.Tensor(sample).long().to(self.device)
                elif dt == 'synergy':
                    batch_dict['synergy'] = torch.Tensor(sample).float().to(self.device)
                else:
                    if sample is not None:
                        batch_dict[dt] = torch.Tensor(sample).float().to(self.device)
                    else:
                        batch_dict[dt] = None


            # forward
            chemical_list = self.chemical + [ batch_dict['chemical'] ]
            dose_list = self.dose + [ batch_dict['dose'] ]
            if self.use_onehot == True:
                onehot_list = [ batch_dict['onehot'] ]
            else:
                onehot_list = []
            data_list = [chemical_list, dose_list, onehot_list]
            y_pred, y2_pred, node_loss, xsg_dict = model(data_list)
            
            # calculate loss
            y = batch_dict['synergy']
            y2 = batch_dict['region']
            loss1 = self.loss_fn(y_pred, y.reshape(-1,1))
            if self.use_auxi == True:
                loss2 = self.loss_fn2(y2_pred, y2)
                loss = loss1 + loss2 + node_loss
            else:
                loss = loss1 + node_loss
            

            # add regularization
            l_lambda = 0.001
            l_norm = sum(p.pow(2.0).sum() for p in model.parameters()) # L2
            # l_norm = sum(p.abs().sum() for p in model.parameters()) #L1
            loss = loss + l_lambda * l_norm
            
            # backward
            loss.backward()


            # optimize
            self.optimizer.step()
            train_loss += loss.item()

            # append to list
            if self.task == 'classification':
                y_pred = torch.sigmoid(y_pred)
            true_list.append( y.cpu().detach().numpy() )
            pred_list.append( y_pred.cpu().detach().numpy().reshape(-1) )

        # end of loop
        train_loss = train_loss / len(btz_list)
        # calculate score
        true_list = np.concatenate(true_list)
        pred_list = np.concatenate(pred_list)
        mask = ~np.isnan(pred_list)
        true_list = true_list[mask]
        pred_list = pred_list[mask]
        if len(true_list)>1 and len(pred_list)>1:
            if self.task == 'classification':
                train_score = skmts.average_precision_score(true_list, pred_list)
            elif self.task == 'regression':
                train_score, pval = scistat.pearsonr(true_list, pred_list)
            else:
                raise ValueError(f'{self.task} not supported!!!')
        else:
            print(f'WARNING, y_pred all nan!!!')
            if self.task == 'classification':
                train_score = 0 # AUPRC
            else:
                train_score = 0 # PCC
        
        self.scheduler.step()

        # validating
        model.eval()
        n_data = valid_data.get_length()
        btz_list = valid_data.generate_batch(batch_size=self.batch_size)
        print(f'validation mode')
        print(f'total data={n_data} | batch_size={self.batch_size} | #batches={len(btz_list)}')
        valid_loss = 0.0
        true_list, pred_list = [], []
        with torch.no_grad():
            for i in tqdm( range(0, len(btz_list)) ):
                # get batch data & set device
                sample_dict = valid_data.get_slice(btz_list[i])
                n_subject = len(btz_list[i])

                batch_dict = {}
                for dt, sample in sample_dict.items():
                    if dt == 'region':
                        batch_dict['region'] = torch.Tensor(sample).long().to(self.device)
                    elif dt == 'synergy':
                        batch_dict['synergy'] = torch.Tensor(sample).float().to(self.device)
                    else:
                        if sample is not None:
                            batch_dict[dt] = torch.Tensor(sample).float().to(self.device)
                        else:
                            batch_dict[dt] = None

                # forward 
                chemical_list = self.chemical + [ batch_dict['chemical'] ]
                dose_list = self.dose + [ batch_dict['dose'] ]
                if self.use_onehot == True:
                    onehot_list = [ batch_dict['onehot'] ]
                else:
                    onehot_list = []
                data_list = [chemical_list, dose_list, onehot_list]
                y_pred, y2_pred, node_loss, xsg_dict = model(data_list)
                
                # calculate loss 
                y = batch_dict['synergy']
                y2 = batch_dict['region']
                loss1 = self.loss_fn(y_pred, y.reshape(-1,1))
                if self.use_auxi == True:
                    loss2 = self.loss_fn2(y2_pred, y2)
                    loss = loss1 + loss2 + node_loss
                else:
                    loss = loss1  + node_loss
                
                # add regularization
                    l_lambda = 0.001
                    l_norm = sum(p.pow(2.0).sum() for p in model.parameters()) # L2
                    #l_norm = sum(p.abs().sum() for p in model.parameters()) # L1
                loss = loss + l_lambda * l_norm
                valid_loss += loss.item()

                # append to list
                if self.task == 'classification':
                    y_pred = torch.sigmoid(y_pred)
                true_list.append( y.cpu().detach().numpy() )
                pred_list.append( y_pred.cpu().detach().numpy().reshape(-1) )
                          
        # end of loop
        valid_loss = valid_loss / len(btz_list)
        # calculate score
        true_list = np.concatenate(true_list)
        pred_list = np.concatenate(pred_list)
        mask = ~np.isnan(pred_list)
        true_list = true_list[mask]
        pred_list = pred_list[mask]
        if len(true_list)>1 and len(pred_list)>1:
            if self.task == 'classification':
                valid_score = skmts.average_precision_score(true_list, pred_list)
            elif self.task == 'regression':
                valid_score, pval = scistat.pearsonr(true_list, pred_list)
            else:
                raise ValueError(f'{self.task} not supported!!!')
        else:
            print(f'WARN, y_pred all nan!!!')
            if self.task == 'classification':
                valid_score = 0 # AUPRC
            else:
                valid_score = 0 # PCC
        return model, train_loss, valid_loss, train_score, valid_score

    def predict(self, model, test_data):
        model.eval()
        n_data = test_data.get_length()
        btz_list = test_data.generate_batch(batch_size=self.batch_size)
        print(f'Make Inference on an indepedent testset')
        print(f'total data={n_data} | batch_size={self.batch_size} | #batches={len(btz_list)}')
        y_list, y2_list, pred_list, pred2_list, chemical_xsg_list, dose_xsg_list = [], [], [], [], [], []
        dose_attn_list, chemical_attn_list = [], []
        emb_dict = {'chemical':None, 'dose':None}
        with torch.no_grad():
            for i in tqdm( range(0, len(btz_list)) ):
                # get batch data & set device
                sample_dict = test_data.get_slice(btz_list[i])
                n_subject = len(btz_list[i])

                batch_dict = {}
                for dt, sample in sample_dict.items():
                    if dt == 'region':
                        batch_dict['region'] = torch.Tensor(sample).long().to(self.device)
                    elif dt == 'synergy':
                        batch_dict['synergy'] = torch.Tensor(sample).float().to(self.device)
                    else:
                        if sample is not None:
                            batch_dict[dt] = torch.Tensor(sample).float().to(self.device)
                        else:
                            batch_dict[dt] = None

                # forward
                chemical_list = self.chemical + [ batch_dict['chemical'] ]
                dose_list = self.dose + [ batch_dict['dose'] ]
                if self.use_onehot == True:
                    onehot_list = [ batch_dict['onehot'] ]
                else:
                    onehot_list = []
                data_list = [chemical_list, dose_list, onehot_list]
                y_pred, y2_pred, node_loss, xsg_dict = model(data_list)
                if self.task == 'classification':
                    y_pred = torch.sigmoid(y_pred)



                # append to list
                if self.use_auxi == True:
                    y2_pred = torch.argmax(y2_pred, 1)
                    y2 = batch_dict['region']
                    y2_list.append( y2.cpu().detach().numpy() )
                    pred2_list.append( y2_pred.cpu().detach().numpy().reshape(-1) )

            
                # append to list
                y = batch_dict['synergy']
                y_list.append( y.cpu().detach().numpy() )
                pred_list.append( y_pred.cpu().detach().numpy().reshape(-1) )
                
                # store subgraph embeddings
                dose_xsg_list.append( xsg_dict['dose'][0].cpu().detach().numpy() )
                chemical_xsg_list.append( xsg_dict['chemical'][0].cpu().detach().numpy() )

                # store attention weights
                dose_attn_list.append( xsg_dict['dose'][1].cpu().detach().numpy() )
                chemical_attn_list.append( xsg_dict['chemical'][1].cpu().detach().numpy() )


        # end of loop
        # collect all subgraph embeddings
        dose_xsg_list = np.concatenate(dose_xsg_list)
        chemical_xsg_list = np.concatenate(chemical_xsg_list)
        # collect all subgraph attnetion weights
        dose_attn_list = np.concatenate(dose_attn_list)
        chemical_attn_list = np.concatenate(chemical_attn_list)
        emb_dict['dose'] = (dose_xsg_list, dose_attn_list)
        emb_dict['chemical'] = (chemical_xsg_list, chemical_attn_list)
    
        # collect ys
        if self.use_auxi == True:
            y2_list = np.concatenate(y2_list)
            pred2_list = np.concatenate(pred2_list)
        else:
            y2_list, pred2_list = [], []
        y_list = np.concatenate(y_list)
        pred_list = np.concatenate(pred_list)
        y_dict = {'y':y_list, 'y_pred':pred_list,
                  'y2':y2_list, 'y2_pred':pred2_list}
        
        return y_dict, emb_dict


    def fit(self, model, train_data):
        # training
        model.train()
        n_data = train_data.get_length()
        btz_list = train_data.generate_batch(batch_size=self.batch_size)
        print(f'training mode')
        print(f'total data={n_data} | batch_size={self.batch_size} | #batches={len(btz_list)}')
        train_loss = 0.0
        true_list, pred_list = [], []
        for i in tqdm( range(0, len(btz_list)) ):
            # zero the parameter gradients
            self.optimizer.zero_grad()

            # get batch data & set device
            sample_dict = train_data.get_slice(btz_list[i])
            n_subject = len(btz_list[i])

            batch_dict = {}
            for dt, sample in sample_dict.items():
                if dt == 'region':
                    batch_dict['region'] = torch.Tensor(sample).long().to(self.device)
                elif dt == 'synergy':
                    batch_dict['synergy'] = torch.Tensor(sample).float().to(self.device)
                else:
                    if sample is not None:
                        batch_dict[dt] = torch.Tensor(sample).float().to(self.device)
                    else:
                        batch_dict[dt] = None


            # forward
            chemical_list = self.chemical + [ batch_dict['chemical'] ]
            dose_list = self.dose + [ batch_dict['dose'] ]
            if self.use_onehot == True:
                onehot_list = [ batch_dict['onehot'] ]
            else:
                onehot_list = []
            data_list = [chemical_list, dose_list, onehot_list]
            y_pred, y2_pred, node_loss, xsg_dict = model(data_list)

            # calculate loss
            y = batch_dict['synergy']
            y2 = batch_dict['region']
            loss1 = self.loss_fn(y_pred, y.reshape(-1,1))
            if self.use_auxi == True:
                loss2 = self.loss_fn2(y2_pred, y2)
                loss = loss1 + loss2 + node_loss
            else:
                loss = loss1 + node_loss

            # add regularization
            l_lambda = 0.001
            l_norm = sum(p.pow(2.0).sum() for p in model.parameters()) # L2
            # l_norm = sum(p.abs().sum() for p in model.parameters()) #L1
            loss = loss + l_lambda * l_norm

            # backward
            loss.backward()


            # optimize
            self.optimizer.step()
            train_loss += loss.item()

            # append to list
            if self.task == 'classification':
                y_pred = torch.sigmoid(y_pred)
            true_list.append( y.cpu().detach().numpy() )
            pred_list.append( y_pred.cpu().detach().numpy().reshape(-1) )

        # end of loop
        train_loss = train_loss / len(btz_list)
        # calculate score
        true_list = np.concatenate(true_list)
        pred_list = np.concatenate(pred_list)
        mask = ~np.isnan(pred_list)
        true_list = true_list[mask]
        pred_list = pred_list[mask]
        if len(true_list)>1 and len(pred_list)>1:
            if self.task == 'classification':
                train_score = skmts.average_precision_score(true_list, pred_list)
            elif self.task == 'regression':
                train_score, pval = scistat.pearsonr(true_list, pred_list)
            else:
                raise ValueError(f'{self.task} not supported!!!')
        else:
            print(f'WARNING, y_pred all nan!!!')
            if self.task == 'classification':
                train_score = 0 # AUPRC
            else:
                train_score = 0 # PCC
        self.scheduler.step()
        return model, train_loss, train_score

