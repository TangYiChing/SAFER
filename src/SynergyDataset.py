"""
create torch's dataset for DrugComb
"""

import numpy as np
import pandas as pd
import pickle as pkl
import torch

import sklearn.feature_extraction as skfet
import sklearn.preprocessing as skpre

class DrugCombDataset:
    """
    :param summary_data: drug combination data,
                         headers=['triplet', 'icombo', 'ipair', 'cid', 'synergy_label', 'region_label']
    :param chemical_subgraph: dataframe with kmer by icombo
    :param dose_subgraph: dataframe with drug by icombo
    """
    def __init__(self, summary_data, chemical_subgraph, dose_subgraph, onehot_subgraph, shuffle=False):
        # load data
        if shuffle:
            self.summary_df = summary_data.sample(frac=1).reset_index(drop=True)
        else:
            self.summary_df = summary_data

        subgraph_dict = {'chemical': chemical_subgraph, 'dose': dose_subgraph, 'onehot': onehot_subgraph}

        # set data
        self.chemical_df = subgraph_dict['chemical']
        self.dose_df = subgraph_dict['dose']
        self.onehot_df = subgraph_dict['onehot']

    def get_length(self):
        return len(self.summary_df)

    def generate_batch(self, batch_size):
        """return a list of index with size=batch_size"""
        n_btz = int(self.get_length() / batch_size)
        if self.get_length() % batch_size != 0:
            n_btz += 1
        slices = np.split( np.arange(n_btz*batch_size), n_btz )
        slices[-1] = slices[-1][:self.get_length() - batch_size * (n_btz-1)]
      
        if len(slices[-1]) == 1:
            slices[-2] = np.concatenate([slices[-2], slices[-1]]) # merge to the second-to-last
            slices = slices[:-1] # and then drop last when size=1
        return slices

    def get_slice(self, idx_list):
        """return sample dictionary of given index"""
        sample_dict = {'chemical': None,
                       'dose': None,
                       'onehot': None,
                       'synergy':None,
                       'region': None}
        # collect data
        icombo = self.summary_df.iloc[idx_list, 1]
        synergy = self.summary_df.iloc[idx_list, 4]
        region = self.summary_df.iloc[idx_list, 5]

        # get labels
        sample_dict['synergy'] = synergy.values
        sample_dict['region'] = region.values

        
        # get subgraph: subject as hyperedges by nodes
        if self.chemical_df is not None:
            sample_dict['chemical'] = self.chemical_df.loc[:, icombo].T.values
        if self.dose_df is not None:
            sample_dict['dose'] = self.dose_df.loc[:, icombo].T.values
        if self.onehot_df is not None:
            sample_dict['onehot'] = self.onehot_df.loc[icombo].values
        
        # check data
        #for dt, sample in sample_dict.items():
        #    print(f'{dt}: {sample}')
        return sample_dict
