"""
"""

import json
import pathlib
import requests
import numpy as np
import pandas as pd

class ParseDrugComb:
    """
    """
    def __init__(self, root=None, debug=False):
        self.debug = debug
        if root is None:
            raise Exception(f'Error, file path is required!!!')
        else:
            if pathlib.Path(root).is_file():
                self.root = root
                # read file
                col_list = ['block_id', 'drug_row', 'drug_col', 'cell_line_name', 'study_name', 'tissue_name',
                    'ri_row', 'ri_col','ic50_row', 'ic50_col', 'drug_row_clinical_phase', 'drug_col_clinical_phase']
                self.df = pd.read_csv(self.root, header=0, sep=",", low_memory=False, usecols=col_list)
            else:
                raise Exception(f'Error, file={root} not found!!!')

    def api2df(self, url, debug=False):
        content = requests.get(url, verify=False)
        data_json = json.loads(content.text)
        data_df = pd.DataFrame.from_dict(data_json)

        if debug:
            #if len(data_json[0]) != len(data_df):
            #    raise Exception(f'Error, number of records is not equal!!!')
            #else:
            print(f'retrieve data from api={url}')
            print(data_df.head())
        return data_df

    def clean_response_data(self, debug=False):
        """
        query limit: at most 1000 blocks per query
        """
        df_list = [] # a list of response table
        interval = 1000
        end = 1469182
        for i in range(1, end, interval):
            url = f'https://api.drugcomb.org/response?from={i}&to={i+interval-1}'
            df = self.api2df(url)
            if len(df) > 0:
                df_list.append(df)
        block_df = pd.concat(df_list, axis=0)
        if self.debug:
            print(f'#retrieve #response table={end}, size={len(df)}')
        block_df.to_pickle('dose_response.pkl')    
        return block_df
 
    def clean_summary_data(self, df, debug=False):
        """
        exclude data that are
        1. monotherapies
        2. non-cancer samples
        3. missing expression data
        """
        # retrieve anti-cancer combination therapies from summary file
        # note, synergy_loewe in summary data has been averaged by block_id
        n_ori = len(df)

        # exclude data
        df = self._remove_monotherapy(df)
        df = self._remove_nonCancer(df)
        #df = self._remove_study(df)
        #print(df)
        
        if debug:
            print(f'excluding number of data={n_ori-len(df)}')
            print(f'clean data={len(df)}')
            #print(df.head())
        return df    

    def clean_cell_data(self, debug=False):
        """
        excludes cell lines with NA/BeatAML expression data
        """
        df = self.api2df('https://api.drugcomb.org/cell_lines')
        n_ori = len(df['name'].unique())

        df = df[~df['expression_data'].isin(['NA', 'BeatAML'])]
        df = df[['name', 'depmap_id', 'ccle_name', 'synonyms']]

        if debug:
            print(f'excluding number of data={n_ori-len(df)}')
            print(f'clean data={len(df)}')
            #print(df.head())
        return df

    def clean_drug_data(self, debug=False):
        """
        exclude drugs with NA/Antibody SMILE
        unify dname with the same SMILE
        """
        df = self.api2df('https://api.drugcomb.org/drugs')
        df = df[df['smiles'] != 'NULL']
        df = df[df['smiles']!='-666']
        df = df[~df['smiles'].str.contains('Antibody')]
        df = df[['dname', 'smiles', 'cid_m', 'cid_s', 'drugbank_id', 'cid', 'inchikey']]

        # check duplicated smile
        dname_dup_dict = {} # dname: [dname with the same SMILE]
        smile_dict = {} # {SMILE: dname}
        for drug in df['dname'].unique():
            # get SMILE
            smile = df[df['dname']==drug]['smiles'].values[0]
            if ';' in smile:
                smile = smile.split(';')[0]
            # check duplicates
            if not smile in smile_dict:
                smile_dict[smile] = drug
            else:
                if not drug in dname_dup_dict:
                    dname_dup_dict[drug] = []
                    dname_dup_dict[drug].append( smile_dict[smile] )
                else:
                    dname_dup_dict[drug].append( smile_dict[smile] )
            #print(f'Found Duplicates={smile} in drug={drug}')
            #print(f'the same as drug={smile_dict[smile]}')

        # modify dname and remove duplicates
        keep_first_dict = {}
        for k, v in dname_dup_dict.items():
            keep_first_dict[k] = v[0]
        df.loc[:, 'dname'] = df.loc[:, 'dname'].replace(to_replace=dname_dup_dict)
        df = df.drop_duplicates(subset=['dname', 'smiles'], keep='first')
        
        if debug:
            for k, v in dname_dup_dict.items():
                if len(v) > 1:
                    print(f'dname={k} has {len(v)} duplicated name(s)')
            print(f'#dname with duplicate SMILE={len(dname_dup_dict)}')

        return df, keep_first_dict

    def _remove_monotherapy(self, df):
        df = df[pd.notnull(df['drug_col'])]
        df = df[pd.notnull(df['drug_row'])]
        return df

    def _remove_nonCancer(self, df):
        df = df[~df['tissue_name'].isin(['malaria'])]
        df = df[~df['study_name'].isin(['NCATS_SARS-COV-2DPI'])]
        return df

    def _remove_study(self, df):
        return df[~df['study_name'].isin(['ASTRAZENECA'])]

    def _remove_naLOEWE(self, df):
        df['synergy_loewe'] = df['synergy_loewe'].replace('\\N', np.nan)
        df = df.dropna(how='any', axis=0)
        return df

    def get_processed_data(self, debug=False):
        """return summary, drug, cell, response data"""
        # retrieve data
        summary_df = self.clean_summary_data(self.df)
        cell_df = self.clean_cell_data()
        drug_df, dname_dup_dict = self.clean_drug_data()
        dose_df = pd.read_pickle('/home/DrugCombHypergraph/data/required_files/dose_response.pkl') #self.clean_response_data()

        # replace dname with same SMILE
        summary_df.loc[:, 'drug_row'] = summary_df.loc[:, 'drug_row'].replace(to_replace=dname_dup_dict)
        summary_df.loc[:, 'drug_col'] = summary_df.loc[:, 'drug_col'].replace(to_replace=dname_dup_dict)

        # obtain combination therapies
        col_list = ['block_id', 'drug_row', 'drug_col', 'cell_line_name', 'study_name', 'tissue_name', 
                    'ri_row', 'ri_col','ic50_row', 'ic50_col', 'drug_row_clinical_phase', 'drug_col_clinical_phase']
        summary_df = summary_df[col_list]
        combinationtherapy = (summary_df['drug_col'].notnull()) & (summary_df['drug_row'].notnull())
        combo_df = summary_df[combinationtherapy]

        # merge summary with dose response matrix
        # use synergy measures from dose response, instead of the averaged ones from summary data
        common_bid = sorted(set(dose_df['block_id']) & set(combo_df['block_id']))
        dose_df = dose_df[dose_df['block_id'].isin(common_bid)]
        df = dose_df.merge(combo_df, on='block_id', how='left')
        
        # remove full agreement synergy but no inhibition effects
        error = (df['inhibition']<0) & (df['synergy_loewe']>0) & (df['synergy_hsa']>0) & (df['synergy_zip']>0) &  (df['synergy_bliss']>0)
        df = df[~error]
       
        # remove additive samples
        syn = (df['synergy_loewe']>10) & (df['synergy_hsa']>10) & (df['synergy_zip']>10) &  (df['synergy_bliss']>10)
        ant = (df['synergy_loewe']<-10) & (df['synergy_hsa']<-10) & (df['synergy_zip']<-10) &  (df['synergy_bliss']<-10)
        syn_df = df[syn]
        ant_df = df[ant]
        qualified_df = pd.concat([syn_df, ant_df], axis=0)

        # average replicates
        dup_list = [] # replicates
        sig_list = [] # single experiemnt
        grps = qualified_df.groupby(['drug_row', 'drug_col', 'cell_line_name'])
        idx_list = grps.groups.keys()
        for idx in idx_list:
            bid_list = sorted(grps.get_group(idx).copy()['block_id'].unique())
            df = qualified_df[qualified_df['block_id'].isin(bid_list)]
            if len(bid_list) > 1:
                avg = df.groupby(['drug_row', 'drug_col', 'cell_line_name','conc_r', 'conc_c', 'tissue_name', 'study_name', 'drug_row_clinical_phase', 'drug_col_clinical_phase']).mean().reset_index()
                dup_list.append(avg)
            else:
                sig_list.append(df)

        sig_df = pd.concat(sig_list, axis=0)
        dup_df = pd.concat(dup_list, axis=0)
        # merge
        col_list = ['drug_row', 'drug_col', 'cell_line_name', 'conc_r', 'conc_c', 'inhibition', 'synergy_zip', 'synergy_loewe', 'synergy_hsa', 'synergy_bliss',
                   'ic50_row', 'ic50_col', 'ri_row', 'ri_col', 'tissue_name', 'study_name', 'drug_row_clinical_phase', 'drug_col_clinical_phase']
        dup_df = dup_df[col_list]
        sig_df = sig_df[col_list]
        df = pd.concat([dup_df, sig_df], axis=0)

        if self.debug:
            n_pairs = df.drop_duplicates(subset=['drug_row', 'drug_col']).shape[0]
            n_triplets = df.drop_duplicates(subset=['drug_row','drug_col', 'cell_line_name']).shape[0]
            print(f'#drugs={len(drug_list)}')
            print(f'#cells={len(cell_list)}')
            print(f'#drug pairs={n_pairs}')
            print(f'#drug-drug-cell triplets={n_triplets}')
            print(f'#data size={len(df)}')
            print(df)
            print(drug_df)
            print(cell_df)
            
        return df, drug_df, cell_df
