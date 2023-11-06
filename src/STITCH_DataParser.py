"""
Parse STITCH's drug target dataset

"""

import pathlib
import numpy as np
import pandas as pd

class ParseSTITCH:
    """
    """
    def __init__(self, root=None , debug=False):
        self.debug = debug
       
        f_dict = {'chemical_link': '9606.protein_chemical.links.v5.0.tsv.gz',
                  'action_type': '9606.actions.v5.0.tsv.gz',
                  'protein_info': '9606.protein.info.v11.5.txt.gz'}
        
        # set root
        if root is None:
            raise Exception(f'Error, file path is required!!!')
        else:
            for k, f in f_dict.items():
                fname_str = root + '/' + f
         
                if pathlib.Path(fname_str).is_file():
                    f_dict[k] = fname_str
                else:
                    raise Exception(f'Error, file={fname_str} not found!!!')
        self.f_dict = f_dict

    def get_clean_action(self):
        """
        exclusion by action type
        """
        df = pd.read_csv(self.f_dict['action_type'], header=0, sep="\t", compression='gzip', engine='python',
                         usecols=['item_id_a', 'item_id_b', 'action'])
        action_list = ['activation', 'inhibition']
        df = df[df['action'].isin(action_list)]
        df = df.drop_duplicates(subset=['item_id_a', 'item_id_b'])
        df.columns = ['cid', 'pid', 'action']
        return df

    def get_clean_proteinInfo(self):
        df = pd.read_csv(self.f_dict['protein_info'], header=0, sep="\t", compression='gzip', engine='python', 
                         usecols=['#string_protein_id', 'preferred_name'])
        df = df.drop_duplicates(subset=['#string_protein_id', 'preferred_name'])
        df.columns = ['pid', 'symbol']
        return df
     
    def get_clean_link(self):
        df = pd.read_csv(self.f_dict['chemical_link'], header=0, sep="\t", compression='gzip', engine='python',
                         usecols=['chemical', 'protein'])
        df = df.drop_duplicates(subset=['chemical', 'protein'])
        df.columns = ['cid', 'pid']
        return df

    def get_processed_data(self):
        chemical_protein_df = self.get_clean_link()
        action_df = self.get_clean_action()
        proteinID_name_df = self.get_clean_proteinInfo()

        df = pd.merge(chemical_protein_df, proteinID_name_df, on='pid', how='left')
        df = df.dropna(how='any', axis=0)

        df = pd.merge(df, action_df, on=['cid','pid'], how='left')
        df = df.dropna(how='any', axis=0)

        if self.debug:
            avg = df['pid'].value_counts().mean()
            print(f'avg target per chemical={avg}')
        return df
