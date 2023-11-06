"""
Select Drug pairs with high variable synergy across tissues

qualified drug pairs:
    1. number of tissues > 3
    2. number of cell lines > 5
    3. coefficient of variation across tissue (i.e., absolute CV) > 1
"""

import argparse
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid", font_scale=1.5)

class Drugpairs:
    def __init__(self, root=None, n_tissue=3, n_cell=5, dose_region='all', debug=False):
        # setting
        self.root = root
        self.n_tissue = int(n_tissue)
        self.n_cell = int(n_cell)
        self.dose_region = dose_region
        self.debug = debug

        # load data
        if self.root is None:
            raise Exception(f'Error, file drugcomb_processed_summary.pkl is required!!!')
        else:
            if pathlib.Path(self.root).is_file():
                self.root = pd.read_pickle(self.root)
            else:
                raise Exception(f'Error, file={self.root} not found!!!')

        # choose dose region
        if self.dose_region != 'all':
            self.root = self.root[self.root['dose_region']==self.dose_region]

    def get_processed_data(self):
        """return qualified summary data"""
        # 1. obtain qualified drug pairs
        data_dict = self._get_ipair_tissue_dict(self.root, self.n_tissue, self.n_cell)
        
        # 2. obtain corresponding icombos
        cv_df, tsp_list, nontsp_list = self._get_cv_table(self.root, data_dict)
        
        # 3. return summary data
        tsp_df = self._get_summary_data(self.root, tsp_list)

        if self.debug:
            print(f'summary stats of tissue-specific drug combination data')
            n_icombo = len(tsp_df['icombo'].unique())
            n_ipair = len(tsp_df['ipair'].unique())
            n_tissue = len(tsp_df['tissue_name'].unique())
            n_cid = len(tsp_df['cid'].unique())
            n_study = len(tsp_df['study_name'].unique())
            print(f'    #icombo={n_icombo}')
            print(f'    #ipairs={n_ipair}')
            print(f'    #tissue={n_tissue}')
            print(f'    #cell={n_cid}')
            print(f'    #study={n_study}')
        return tsp_df

    def _get_ipair_tissue_dict(self, df, n_tissue, n_cell):
        """
        :param df: dataframe with headers=['drug_row', 'drug_col', 'cell_line_name', 'ipair', 'tissue_name']
        :param n_tissue: integer representing threshold of number of tissues
        :param n_cell: integer representing threshold of number of cells
        :retur data_dict: dictionary with key=ipair, value=tissue_list
        """
        # remove duplicates
        df = df.drop_duplicates(subset=['drug_row', 'drug_col', 'cell_line_name'], keep='first')

        # collect results
        data_dict = {} #ipair:tissue_list

        # loop through ipair 
        count = df.drop_duplicates(subset=['ipair', 'tissue_name'], keep='first').groupby('ipair')['tissue_name'].count().to_frame(name='tissue count')
        ipair_list = count[count['tissue count']>=n_tissue].index.tolist()
        for ipair in ipair_list:
            tissue_list = []
            for tissue in df[df['ipair']==ipair]['tissue_name'].unique():
                cell_list = df[(df['ipair']==ipair)&(df['tissue_name']==tissue)]['cid'].unique()
                if len(cell_list) >= n_cell:
                    tissue_list.append(tissue)
            # finalize
            if len(tissue_list) >= n_tissue:
                data_dict[ipair] = tissue_list
        return data_dict

    def _get_cv_table(self, df, ipair_tissue_dict):
        """
        :param df: dataframe with headers=['drug_row', 'drug_col', 'cell_line_name', 'ipair', 'tissue_name', 'synergy_loewe']
        :param ipair_tissue_dict: dictionary with key=ipair, value=tissue_list
        :return cv_df: dataframe with headers=['ipair', 'num.tissues', 'avg.cells per tissue', 
                                               'avg.synergy', 'std.synergy', 'coefficient of variation', 
                                               'tissue specific']
        """
        # collect results
        tsp_list = [] # icombo
        nontsp_list = [] # icombo
        record_list = []
        for ipair, tissue_list in ipair_tissue_dict.items():
            t = df[(df['ipair']==ipair)&(df['tissue_name'].isin(tissue_list))]
            icombo_list = sorted(t['icombo'].unique())
            num_tissue = len(tissue_list)
            num_cell_per_tissue = t.groupby(['ipair', 'tissue_name'])['cid'].count().mean()
            avg_synergy = t['synergy_loewe'].mean()
            std_synergy = t['synergy_loewe'].std()
            cv = std_synergy / avg_synergy # calculate coefficient of variation
            if abs(cv) > 1:
                tsp = 'Yes'
                tsp_list += icombo_list
            else:
                tsp = 'No'
                nontsp_list += icombo_list
            record = (ipair, num_tissue, num_cell_per_tissue, avg_synergy, std_synergy, cv, tsp)
            record_list.append(record)
        col_list = ['ipair', 'num.tissues', 'avg.cells per tissue', 'avg.synergy', 'std.synergy', 'coefficient of variation', 'tissue specific']
        cv_df = pd.DataFrame.from_records(record_list, columns=col_list)

        if self.debug:
            n_yes = len(cv_df[cv_df['tissue specific']=='Yes'])
            n_no = len(cv_df[cv_df['tissue specific']=='No'])
            print(f'threshold for n_tissue={self.n_tissue} | n_cell={self.n_cell}')
            print(f'high variable drug pairs (coefficient of variation > 1): #tsp={n_yes} | #non-tsp={n_no}')
        return cv_df, tsp_list, nontsp_list

    def _get_summary_data(self, df, icombo_list):
        """
        :param df: dataframe with headers=['icombo']
        :param icombo_list: list of icombo
        :return summary_df: dataframe of summary data
        """
        return df[df['icombo'].isin(icombo_list)]
