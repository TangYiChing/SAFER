"""
miscellaneous functions
"""

import torch
import pathlib
import pickle as pkl
#import cupy as cp
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style
import sklearn.feature_extraction as skfet
import sklearn.metrics as skmts
import scipy.stats as scistat
style.use('fivethirtyeight')

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def eval_mts(y_list, y_pred_list, task_str):
    """return evaluation  metrices"""
    mask = ~np.isnan(y_pred_list)
    y_list = y_list[mask]
    y_pred_list = y_pred_list[mask]
    if task_str == 'classification':
        auc = skmts.roc_auc_score(y_list, y_pred_list, average='micro')
        aucprc = skmts.average_precision_score(y_list, y_pred_list, average='micro')
        y_pred_list = (y_pred_list>0.5).astype(int)
        acc = skmts.accuracy_score(y_list, y_pred_list)
        mcc = skmts.matthews_corrcoef(y_list, y_pred_list)
        f1 = skmts.f1_score(y_list, y_pred_list, average='micro')
        precision = skmts.precision_score(y_list, y_pred_list)
        recall = skmts.recall_score(y_list, y_pred_list)
        kappa = skmts.cohen_kappa_score(y_list, y_pred_list)
        balanced_acc = skmts.balanced_accuracy_score(y_list, y_pred_list)
       
        df = pd.DataFrame({'metrices': ['AUC', 'AUPRC', 'Accuracy', 'MCC', 'F1', 'Precision', 'Recall', 'Kappa', 'Balanced_Accuracy'],
                           'score': [auc, aucprc, acc, mcc, f1, precision, recall, kappa, balanced_acc]})

        tn, fp, fn, tp = skmts.confusion_matrix(y_list, y_pred_list).ravel()
        print(f'confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}')

    elif task_str == 'regression':
        mae = skmts.mean_absolute_error(y_list, y_pred_list)
        mse = skmts.mean_squared_error(y_list, y_pred_list)
        rmse = skmts.mean_squared_error(y_list, y_pred_list, squared=False)
        r2 = skmts.r2_score(y_list, y_pred_list)
        pcc, pval = scistat.pearsonr(y_list, y_pred_list)
        spr, pval = scistat.spearmanr(y_list, y_pred_list)

        df = pd.DataFrame({'metrices': ['MAE', 'MSE', 'RMSE', 'R2', 'Pearson r', 'Spearman r'],
                           'score': [mae, mse, rmse, r2, pcc, spr]})
    else:
        raise ValueError(f'Error, {task_str} should be either classification or regression!!!')
    return df

def get_imratio(df, y_str):
    n_pos = (df[y_str]==1).sum()
    imratio = n_pos/len(df)
    return imratio

def sampling(df, y_str, r=2):
    """return samples
    r = ratio of positives to negatives
    r > 1, return n_pos > n_neg
    r < 1, return n_pos < n_neg
    """
    # get pos, neg
    pos_df = df[df[y_str]==1]
    neg_df = df[df[y_str]==0]

    # define n_samples
    if r > 1:
        if len(pos_df) >= len(neg_df):
            n_neg = len(neg_df)
            n_pos = n_neg * r
            if n_pos <= len(pos_df):
                pos_df = pos_df.sample(n=int(n_pos), replace=False)
            else:
                pos_df = pos_df.sample(n=int(n_pos), replace=True)
        else:
            n_pos = len(pos_df)
            n_neg = n_pos // r
            if n_neg <= len(neg_df):
                neg_df = neg_df.sample(n=int(n_neg), replace=False)
            else:
                neg_df = neg_df.sample(n=int(n_neg), replace=True)
        
    else:
        if len(pos_df) <= len(neg_df):
            n_pos = len(pos_df)
            n_neg = n_pos // r
            if n_neg <= len(neg_df):
                neg_df = neg_df.sample(n=int(n_neg), replace=False)
            else:
                neg_df = neg_df.sample(n=int(n_neg), replace=True)
        else:
            n_neg = len(neg_df)
            n_pos = n_neg * r
            if n_pos <= len(pos_df):
                pos_df = pos_df.sample(n=int(n_pos), replace=False)
            else:
                pos_df = pos_df.sample(n=int(n_pos), replace=True)


    # merge back to df
    df = pd.concat([pos_df, neg_df], axis=0)
    df = df.sample(frac=1).reset_index(drop=True) #shuffle
    
    # display stats
    n_pos = (df[y_str]==1).sum()
    n_neg = (df[y_str]==0).sum()
    print(f'sampling imratio={r}: n_pos={n_pos} | n_neg={n_neg}')
    return df

def sampling_pos(df, y_str, r=0.4):
    """upsample positive samples to have n_minor/n_total=imratio"""
    pos_df = df[df[y_str]==1]
    neg_df = df[df[y_str]==0]
    
    # define n
    n_pos = len(pos_df) * float(r)
    n_neg = len(pos_df) * (1 - float(r))

    # select positive samples
    pos_df = pos_df.sample(n=int(n_pos), replace=False)
    neg_df = neg_df.sample(n=int(n_neg), replace=False)
    df = pd.concat([pos_df, neg_df], axis=0)

    n_pos = (df[y_str]==1).sum()
    n_neg = (df[y_str]==0).sum()
    print(f'after sampling: n_pos={n_pos} | n_neg={n_neg}')
    return df

def plot_LearningCurve(train_log, test_log, ylabel, fout):
    # figure size in inches
    #sns.set(rc={"figure.figsize":(11.7, 8.27)}) #width=11.7, #height=8.27

    # font size
    #sns.set(font_scale=2)

    n_epoch = len(train_log)
    train_df = pd.DataFrame({'epoch':[i+1 for i in range(0,n_epoch)],
                             ylabel:train_log,
                             'data':['train set']*n_epoch})
    test_df = pd.DataFrame({'epoch':[i+1 for i in range(0,n_epoch)],
                            ylabel:test_log,
                            'data':['valid set']*n_epoch})

    df = pd.concat([train_df, test_df], axis=0)
    fig = sns.lineplot(x = "epoch", y = ylabel, hue = "data", data = df)
    plt.savefig(fout+'.LearningCurve.png', bbox_inches='tight', dpi=200)
    return fig

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def transform_H(H):
    """
    """
    transformer = skfet.text.TfidfTransformer()
    H_arr = transformer.fit_transform(H).todense()
    return H_arr

def generate_G_from_H(H, variable_weight=False):
        """
        calculate G from hypgraph incidence matrix H
        :param H: hypergraph incidence matrix H
        :param variable_weight: whether the weight of hyperedge is variable, this is likely useful to penalize very large pathways
        :return: G

        reference: https://github.com/luoyuanlab/SHINE/blob/main/utils/hg_ops.py
        """
        
        H = np.array(H, dtype=float)
        n_edge = H.shape[1]
        W = np.ones(n_edge) # the weight of the hyperedge
        DV = np.sum(H * W, axis=1) # the degree of the node
        DE = np.sum(H, axis=0) # the degree of the hyperedge

        # mask zeros
        zero_idx = (np.abs(DE)==0)
        masked_DE = np.ma.array(DE, mask=zero_idx)

        invDE = np.mat(np.diag(np.ma.power(DE, -1)))
        DV2 = np.mat(np.diag(np.ma.power(DV, -0.5)))
        W = np.mat(np.diag(W))
        H = np.mat(H)
        HT = H.T

        if variable_weight:
            DV2_H = DV2 * H
            invDE_HT_DV2 = invDE * HT * DV2
            return DV2_H, W, invDE_HT_DV2
        else:
            G = DV2 * H * W * invDE * HT * DV2
            #print(f'G: min={G.min().min()} | max={G.max().max()}')
            if np.isnan(G.sum().sum()):
                raise ValueError(f'Error, G has nan!!!')
            return G

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

class RMSELoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.eps = eps

    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss

class MAPELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, yhat, y):
        loss = torch.mean( torch.abs( (y-yhat)/y ) )
        return loss

def loss_fn(output, target):
    # MAPE loss
    return torch.mean(torch.abs((target - output) / target))
