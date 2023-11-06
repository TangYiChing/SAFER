"""
Parse CCLE datasets

download link: https://depmap.org/portal/download/all/
version: Public 22Q4
"""

import pathlib
import numpy as np
import pandas as pd

import sklearn.preprocessing as skpre

class ParseCCLE:
    """
    """
    def __init__(self, root=None, omics=None, model=None, transformed="MinMax", debug=False):
        self.transformed = transformed
        self.debug = debug
        self.omics = omics
                 
        # set root      
        if root is None:
            raise Exception(f'Error, file path is required!!!')
        else:
            if pathlib.Path(root).is_file():
                self.root = root
            else:
                raise Exception(f'Error, file={root} not found!!!')

        
        # set omics
        if self.omics is None:
            raise Exception(f'Error, omics is required, [exp, cnv, mut, ef, es, grn, bgrn, tfgrn, btfgrn, bes, bef]!!!')
 
        elif self.omics in ['exp', 'cnv', 'ef', 'es', 'grn', 'bgrn', 'tfgrn', 'btfgrn', 'bes', 'bef']: # tfgrn: tf-gene regulatory network from GRAND
            # grn, bgrn, tfgrn, btfgrn are data from GRAND, representing tf-gene regulatory network
            # grn: gene by ach_id, computed TF-Gene targeting scores, averaging across TFs
            # bgrn: binarized grn, 1: grn>0 and 0 otherwise
            # tfgrn: tf by ach_id, computed TF-Gene targeting scores, averaging across Genes
            # btfgrn: binarized tfgrn, 1: tfgrn>0 and 0 otherwise
            self.df = pd.read_csv(self.root, header=0, index_col=0)
        elif self.omics == 'mut':
            col_list = ['RefCount', 'AltCount', 'DepMap_ID', 'HugoSymbol']
            self.df = pd.read_csv(self.root, header=0, usecols=col_list, low_memory=False)
        else:
            raise Exception(f'Error, omics={omics} not supported!!!')

        # set model
        if model is None:
            raise Exception(f'Error, file path is required!!!')
        else:
            if pathlib.Path(model).is_file():
                col_list = ['ModelID', 'CCLEName', 'OncotreeLineage', 'OncotreePrimaryDisease']
                self.model = pd.read_csv(model, header=0, usecols=col_list)
            
            else:
                raise Exception(f'Error, file={model} not found!!!')


    def get_processed_data(self):
        # omics data
        if self.omics == 'exp':
            df = self.clean_exp(self.df)
        elif self.omics == 'cnv':
            df = self.clean_cnv(self.df)
        elif self.omics == 'mut':
            df = self.clean_mut(self.df)
        elif self.omics == 'ef':
            df = self.clean_ef(self.df)
        elif self.omics == 'bef':
            df = self.clean_binary_ef(self.df)
        elif self.omics == 'es':
            df = self.clean_es(self.df)
        elif self.omics == 'bes':
            df = self.clean_binary_es(self.df)
        elif self.omics == 'grn':
            df = self.clean_grn(self.df)
        elif self.omics == 'bgrn':
            df = self.clean_binary_grn(self.df)
        elif self.omics == 'tfgrn':
            df = self.clean_tfgrn(self.df)
        elif self.omics == 'btfgrn':
            df = self.clean_binary_tfgrn(self.df)
        else:
            raise Exception(f'Error, omics={self.omics} not supported!!!')

        if np.isnan(df.sum().sum()):
            raise ValueError(f'Error, input contains nan!!!')

        # model annotation
        model = self.clean_model(self.model)
        # subsetting to include cell line with model annotation
        cell_list = sorted(list(set(model['depmap_id']) & set(df.columns)))
        model = model[model['depmap_id'].isin(cell_list)]
        df = df[cell_list]
        return df, model



    def clean_model(self, df):
        """
        note: map TCGA study abbreviation:
        https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/tcga-study-abbreviations
        ACH-000043: non-cancerous fibroblasts
        """
        df.rename(columns={'ModelID':'depmap_id', 'CCLEName':'ccle_name'}, inplace=True)
        name_abb_dict = {'Ovarian Epithelial Tumor':'TCGA_OV', 'Acute Myeloid Leukemia':'TCGA_LAML',
                         'Colorectal Adenocarcinoma':'TCGA_COAD', 'Melanoma':'TCGA_SKCM',
                         'Bladder Urothelial Carcinoma':'TCGA_BLCA', 'Non-Small Cell Lung Cancer':'TCGA_LUSC',
                         'Renal Cell Carcinoma':'TCGA_KIRC', 'Invasive Breast Carcinoma':'TCGA_BRCA',
                         'Pancreatic Adenocarcinoma': 'TCGA_PAAD', 'Non-Hodgkin Lymphoma':'TCGA_DLBC',
                         'Diffuse Glioma':'TCGA_GBM', 'Esophagogastric Adenocarcinoma':'TCGA_STAD',
                         'Prostate Adenocarcinoma': 'TCGA_PRAD', 'Pleural Mesothelioma': 'TCGA_MESO',
                         'Ovarian Germ Cell Tumor': 'TCGA_OV', 'Hepatocellular Carcinoma': 'TCGA_LIHC',   
                         'Rhabdomyosarcoma':'TCGA_SARC', 'Sarcoma, NOS':'TCGA_SARC', 'Ewing Sarcoma':'TCGA_SARC',
                         'Fibrosarcoma':'TCGA_SARC', 'Well-Differentiated Thyroid Cancer':'TCGA_THCA',
                         'Myeloproliferative Neoplasms':'TCGA_LCML',
                         'B-Lymphoblastic Leukemia/Lymphoma':'NA', 'Meningothelial Tumor':'NA',
                         'Embryonal Tumor':'NA', 'Hodgkin Lymphoma':'NA'}
        df.loc[:, 'tcga abbrev'] = df.loc[:, 'OncotreePrimaryDisease'].replace(to_replace=name_abb_dict)

        if self.debug:
            ori_disease = df['OncotreePrimaryDisease'].unique()
            abb_disease = [name for name, abb in name_abb_dict.items() if abb != 'NA'] 
            not_found = list(set(ori_disease)-set(abb_disease))
            print(f'# primary disease={len(ori_disease)}')
            print(f'# of diseases have TCGA Abbreviation={len(abb_disease)}')
            print(f'# of diseases not in TCGA study={len(not_found)}')
            print(f'not yet found\n{not_found}')
            
        return df

    def _transform_data(self, df, method='MinMax'):
        """return transformed data"""
        # choose scaler
        if method == 'Standard':
            scaler = skpre.StandardScaler()
        elif method == 'MinMax':
            scaler = skpre.MinMaxScaler()
        else:
            raise ValueError(f'{method} is not supported, try Standard or MinMax!!!')

        # transform data
        eps = 1e-6
        data_arr = scaler.fit_transform(df.values+eps)
        df = pd.DataFrame(data_arr, columns=df.columns, index=df.index)
        return df

    def clean_grn(self, df):
        """
        return dataframe with gene by sample 
        
        update: MinMax scaling to make sure no negative values
        """
        if df.isnull().sum().sum() > 0:
            df.fillna(0, inplace=True) # fill missing value with 0

        col_list = [col.split(' ')[0] for col in df.columns]
        df.columns = col_list # use Gene Symbol instead of ID

        #if self.transformed is not None:
        #    df = self._transform_data(df, method=self.transformed)
        df = self._transform_data(df, method='MinMax')
                 
        if self.debug:
            print(f'#genes={len(df.T)} |#samples={len(df)}')
        return df.T # gene by sample

    def clean_binary_grn(self, df):
        """
        return dataframe with gene by sample 
        
        """
        if df.isnull().sum().sum() > 0:
            df.fillna(0, inplace=True) # fill missing value with 0

        col_list = [col.split(' ')[0] for col in df.columns]
        df.columns = col_list # use Gene Symbol instead of ID

        # binarize probabilites of tf-targeted genes
        df = df.applymap(lambda x: 1 if x>0 else 0)
        
        if self.debug:
            print(f'#genes={len(df.T)} |#samples={len(df)}')
        return df.T # gene by sample


    def clean_tfgrn(self, df):
        """
        return dataframe with tf by sample

        update: MinMax scaling to make sure no negative values
        """
        if df.isnull().sum().sum() > 0:
            df.fillna(0, inplace=True) # fill missing value with 0

        col_list = [col.split(' ')[0] for col in df.columns]
        df.columns = col_list # use Gene Symbol instead of ID

        #if self.transformed is not None:
        #    df = self._transform_data(df, method=self.transformed)
        df = self._transform_data(df, method='MinMax')

        if self.debug:
            print(f'#TFs={len(df.T)} |#samples={len(df)}')
        return df.T # TF by sample

    def clean_binary_tfgrn(self, df):
        """
        return dataframe with tf by sample
        """
        if df.isnull().sum().sum() > 0:
            df.fillna(0, inplace=True) # fill missing value with 0

        col_list = [col.split(' ')[0] for col in df.columns]
        df.columns = col_list # use Gene Symbol instead of ID

        # binarize probabilites of tf-targeting
        df = df.applymap(lambda x: 1 if x>0 else 0)

        if self.debug:
            print(f'#TFs={len(df.T)} |#samples={len(df)}')
        return df.T # TF by sample

    def clean_exp(self, df):
        """
        return dataframe with gene by sample 
        
        """
        if df.isnull().sum().sum() > 0:
            df.fillna(0, inplace=True) # fill missing value with 0

        col_list = [col.split(' ')[0] for col in df.columns]
        df.columns = col_list # use Gene Symbol instead of ID

        if self.transformed is not None:
            df = self._transform_data(df, method=self.transformed)
        
        if self.debug:
            print(f'#genes={len(df.T)} |#samples={len(df)}')
        return df.T # gene by sample

    def clean_cnv(self, df):
        """
        return dataframe with gene by sample

        """
        if df.isnull().sum().sum() > 0:
            df.fillna(0, inplace=True) # fill missing value with 0

        col_list = [col.split(' ')[0] for col in df.columns]
        df.columns = col_list # use Gene Symbol instead of ID

        if self.transformed is not None:
            df = self._transform_data(df, method=self.transformed)

        if self.debug:
            print(f'#genes={len(df.T)} |#samples={len(df)}')
        return df.T # gene by sample

    def clean_mut(self, df):
        """
        return dataframe with gene by sample

        """
        sub_df = df.groupby(['DepMap_ID', 'HugoSymbol']).agg({'AltCount': 'sum', 'RefCount': 'sum'})
        sub_df['MutationRate'] = sub_df['AltCount']/(sub_df['RefCount']+sub_df['AltCount'])
        sub_df = sub_df.reset_index()
        sub_df = pd.pivot(sub_df, index='HugoSymbol', columns='DepMap_ID', values='MutationRate')
        sub_df[sub_df == np.inf] = 0
        sub_df.fillna(0, inplace=True)

        df = sub_df.rename_axis(index=None)
        df = df.rename_axis(None, axis=1)

        if self.transformed is not None:
            df = self._transform_data(df, method=self.transformed)

        if self.debug:
            print(f'#genes={len(df)} |#samples={len(df.T)}')
            print(f'summary stats of mutation rate per sample\n{df.describe()}')
        return df

    def clean_ef(self, df):
        """
        return dataframe with gene by sample 
        
        update: MinMax scaling to make sure no negative values
        """
        if df.isnull().sum().sum() > 0:
            df.fillna(0, inplace=True) # fill missing value with 0

        col_list = [col.split(' ')[0] for col in df.columns]
        df.columns = col_list # use Gene Symbol instead of ID

        #if self.transformed is not None:
        #    df = self._transform_data(df, method=self.transformed)
        df = self._transform_data(df, method='MinMax')

        if self.debug:
            print(f'#genes={len(df.T)} |#samples={len(df)}')
        return df.T # gene by sample

    def clean_binary_ef(self, df):
        """
        return dataframe with gene by sample

        """
        if df.isnull().sum().sum() > 0:
            df.fillna(0, inplace=True) # fill missing value with 0

        col_list = [col.split(' ')[0] for col in df.columns]
        df.columns = col_list # use Gene Symbol instead of ID

        # binarize probabilites of being non-essential gene distribution
        # ef < -0.5 represents significat essential, while ef < -1 indicates strong killing
        df = df.applymap(lambda x: 1 if x<-0.5 else 0)

        if self.debug:
            print(f'#genes={len(df.T)} |#samples={len(df)}')
        return df.T # gene by sample

    def clean_es(self, df):
        """
        return dataframe with gene by sample

        """
        if df.isnull().sum().sum() > 0:
            df.fillna(0, inplace=True) # fill missing value with 0

        col_list = [col.split(' ')[0] for col in df.columns]
        df.columns = col_list # use Gene Symbol instead of ID

        if self.transformed is not None:
            df = self._transform_data(df, method=self.transformed)

        if self.debug:
            print(f'#genes={len(df.T)} |#samples={len(df)}')
        return df.T # gene by sample


    def clean_binary_es(self, df):
        """
        return dataframe with gene by sample

        """
        if df.isnull().sum().sum() > 0:
            df.fillna(0, inplace=True) # fill missing value with 0

        col_list = [col.split(' ')[0] for col in df.columns]
        df.columns = col_list # use Gene Symbol instead of ID

        # binarize probabilites of being essential genes
        df = df.applymap(lambda x: 1 if x>0.5 else 0)

        if self.debug:
            print(f'#genes={len(df.T)} |#samples={len(df)}')
        return df.T # gene by sample
