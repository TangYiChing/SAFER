"""
Parse Chemical Checker datasets

download link: https://chemicalchecker.org/downloads/root
version: NA
"""

import h5py
import pathlib
import numpy as np
import pandas as pd
import itertools as itr
import DrugComb_DataParser as DrugCombParser

from torch.utils.data import Dataset, DataLoader

class ChemicalCheckerDataset(Dataset):
    def __init__(self, root, pkl):
        """
        :param root: folder contains chemical checker h5 files
        :param pkl:  pickle file of inchikey of drugs
        """
        self.root = root
        self.df = pd.read_pickle(pkl)

        # similarity spaces
        self.ss_list = [''.join(cc) for cc in itr.product(['A','B','C','D','E'], ['1','2','3','4','5']) ]

    def __len__(self):
        return len(self.df)

    def _get_signature(self, key, space):
        with h5py.File(self.root+space+'.h5') as f:
            bkey = bytes(key, "utf-8")
            bkey_arr = f['keys'][:]
            bkey_idx = np.where(bkey_arr==bkey)[0][0]
            sig = f['V'][bkey_idx]
        return sig

    def __getitem__(self, idx):
        sig_list = [] # cc sigature
        key = self.df.iloc[idx]['inchikey']
        for ss in self.ss_list:
            sig = self._get_signature(key, ss)
            sig_list.append(sig)
        cc = np.hstack(sig_list)
        return cc
        
class ParseCC:
    """
    """
    def __init__(self, root=None, debug=False):

        # similarity spaces
        self.ss_list = [''.join(cc) for cc in itr.product(['A','B','C','D','E'], ['1','2','3','4','5']) ]

        # set root      
        if root is None:
            raise Exception(f'Error, file path is required!!!')
        else:
            self.root = root
            for ss in self.ss_list:
                fin = self.root+ss+'.h5'
                if pathlib.Path(fin).is_file():
                    pass
                
                else:
                    raise Exception(f'Error, file={fin} not found!!!')

    def get_signature(self, key, space):
        if self._has_key(key, space):
            # read data
            fin = self.root + space + '.h5'
            with h5py.File(fin) as f:
                bkey = bytes(key, "utf-8")
                bkey_arr = f['keys'][:]
                bkey_idx = np.where(bkey_arr==bkey)[0][0]
                signature = f['V'][bkey_idx]
        else:
            signature = None
        return signature

    def _get_key(self, space):
        if space not in self.ss_list:
            raise ValueError(f'Error, space={space} not found!!!')

        # read data
        fin = self.root + space + '.h5'
        with h5py.File(fin) as f:
            key_arr = f['keys'][:] # array of inChiKey bytes

        # bytes to string
        key_list = [ k.decode("utf-8") for k in key_arr ]
        return key_list # list of inChiKey

    def _has_key(self, key, space):
        if space not in self.ss_list:
            raise ValueError(f'Error, space={space} not found!!!')

        # read data
        fin = self.root + space + '.h5'
        with h5py.File(fin) as f:
            key_arr = f['keys'][:] # array of inChiKey bytes

        # string to bytes
        bkey = bytes(key, "utf-8")
        if bkey in key_arr:
            return True
        else:
            return False

if __name__ == '__main__':
    # load drug data
    root = './'
    if pathlib.Path(root+'/drugcomb_summary.pkl').is_file():
        summary_df = pd.read_pickle(root+'/drugcomb_summary.pkl')
        drug_df = pd.read_pickle(root+'/drugcomb_drug.pkl')
        cell_df = pd.read_pickle(root+'/drugcomb_cell.pkl')
    else:
        drugcomb = srcDrugCombParser.ParseDrugComb(root=data_dict['summary'], debug=args.DEBUG)
        summary_df, drug_df, cell_df = drugcomb.get_processed_data()

    # keep drug with cc dataset
    cc = ParseCC(root='./required_file/')
    ss_list = [''.join(cc) for cc in itr.product(['A','B','C','D','E'], ['1','2','3','4','5']) ]
    keep_list = []
    for inchikey in drug_df['inchikey'].unique():
        n = 0
        for ss in ss_list:
            if cc._has_key(inchikey, ss):
                n+=1
        if n == 25:
            keep_list.append(inchikey)
    keep_df = drug_df[drug_df['inchikey'].isin(keep_list)]
    keep_df.to_pickle('drugcomb_cc_drug.pkl')
    print(f'{}len(keep_list) ({len(keep_df)/len(drug_df)*100}) have cc dataset')
