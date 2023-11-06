"""
Implementation of Hao et al., 2018's method to obtain tissue-specific proteins
Hao, Y., Quinnies, K., Realubit, R., Karan, C., & Tatonetti, N. P. (2018). Tissue‐Specific Analysis of Pharmacological Pathways. CPT: Pharmacometrics & Systems Pharmacology, 7(7), 453–463. https://doi.org/10.1002/psp4.12305

"""

import json
import pathlib
import requests
import numpy as np
import pandas as pd
import scipy.stats as scistat

class ParseGTEx:
    """
    """
    def __init__(self, root=None, debug=False):
        self.debug = debug
        if root is None:
            raise Exception(f'Error, file path to gene_median_tpm.gct.gz required!!!')
        else:
            if pathlib.Path(root).is_file():
                self.root = root
                # read file
                self.df = pd.read_csv(self.root, skiprows=2, header=0, sep='\t', compression='gzip')
            else:
                raise Exception(f'Error, file={root} not found!!!')
    def get_processed_data(self, debug=False):
        """return tissue-specific protein"""
        # retrieve data
        df = self.df.iloc[:, 1:].set_index('Description') # remove EnsembleID and set GeneSymbol as index

        # transform data
        df = self._log_transform(df)

        # median normalization
        df = df.apply(lambda x: ( x - df.median(axis=1) ) / (df.max(axis=1)-df.min(axis=1)) )
        df = df.replace([np.inf, -np.inf], np.nan)
        df.dropna(inplace=True)

        # transform again
        df = self._log_transform(df)

        # compute p-value to determine whether the protein is over-expressed in a tissue
        # threshold for p table is 0.05 (i.e., t1)
        p_df = self._compute_p_protein(df, threshold=0.05)

        # convert to s_protein to determine whetehr the protein is specifically expressed in a tissue
        # threshold for s table is 0 (i.e., t2)
        s_df = self._compute_s_protein(p_df, threshold=0)
        
        # report tissue-specific protein
        df = s_df.mean(axis=1).to_frame(name='tissue-specific protein')
        df = df[df['tissue-specific protein']>0] # new rule: should be supported by at least one tissue
        return df

    def get_s_protein(self):
        """return s_df"""
        # retrieve data
        df = self.df.iloc[:, 1:].set_index('Description') # remove EnsembleID and set GeneSymbol as index

        # transform data
        df = self._log_transform(df)

        # median normalization
        df = df.apply(lambda x: ( x - df.median(axis=1) ) / (df.max(axis=1)-df.min(axis=1)) )
        df = df.replace([np.inf, -np.inf], np.nan)
        df.dropna(inplace=True)

        # transform again
        df = self._log_transform(df)

        # compute p-value to determine whether the protein is over-expressed in a tissue
        # threshold for p table is 0.05 (i.e., t1)
        p_df = self._compute_p_protein(df, threshold=0.05)

        # convert to s_protein to determine whetehr the protein is specifically expressed in a tissue
        # threshold for s table is 0 (i.e., t2)
        s_df = self._compute_s_protein(p_df, threshold=0)
        return s_df

    def _compute_p_protein(self, df, threshold=0.05):
        """
        :param df: dataframe with gene symbol as index and tissue name as columns
        :return p_df: p_protein indicating whether the target protein is over-expresed in a tissue
        """
        p_df = df.copy()
        for tissue in df.columns: # compute zscore and pvalue for each tissue
            data = df[tissue].values
            zscore = scistat.zscore(data)
            pvalue = scistat.norm.pdf( abs(zscore) )*2 # two-sided

            # update p_df
            p_df[tissue] = pvalue

            # convert to 1/0 label
            p_df[tissue] = p_df[tissue].apply(lambda x: 1 if x< float(threshold) else 0)
        return p_df

    def _compute_s_protein(self, p_df, threshold=0):
        """
        :param df: p_protein table of gene by tissue: 1 over-expressed in a tissue, 0 otherwise
        :param s_df: s_protein indicating whether the protein is specifically expression in a tissue
        """
        # convert to s_protein to determine whetehr the protein is specifically expressed in a tissue
        # threshold for s table is 0 (i.e., t2)
        s_df = p_df.apply(lambda x: x/ p_df.sum(axis=1))
        s_df = s_df.fillna(0)
        return s_df

    def _log_transform(self, df):
        """return log-transformed df"""
        # use df+2 to account for zero and extrem values (Hao et al., 2018)
        df = np.log2(df+2)
        return df


