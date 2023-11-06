"""
Parse MSigDB datasets

Download link: https://www.gsea-msigdb.org/gsea/msigdb (C2, C3, C5)
               Homo_sapiens.gene_info.gz (from NCBI's ftp)
keep protein-coding genes only
"""

import numpy as np
import pandas as pd
import pathlib

class ParseMSigDB:
    """
    """
    def __init__(self, root=None, ncbi=None, debug=False):
        self.debug = debug
        self.ncbi = ncbi

        # set root
        if root is None:
            raise Exception(f'Error, file path is required!!!')
        else:
            if pathlib.Path(root).is_file():
                self.root = pd.read_csv(root, header=None, sep=" ")
            else:
                raise Exception(f'Error, file={root} not found!!!')


    def get_processed_data(self):
        df = self._gmt2matrix(self.root)

        # use protein-coding genes only
        #pcg_list = self._keep_pcg()
        #g_list = sorted(set(df.index)&set(pcg_list))
        #df = df.loc[g_list]

        if self.debug:
            node_degree = df.sum(axis=1).to_frame(name='node degree')
            edge_degree = df.sum(axis=0).to_frame(name='edge degree')
            print(f'summary stats of hypergraph')
            print(f'#nodes (genes)={df.shape[0]} | #edges (TFs)={df.shape[1]}')
            print('node degree: median+-std (min, max)={:}+-{:} ({:}, {:})'.format(node_degree['node degree'].median(),
                                                                                   node_degree['node degree'].std(),
                                                                         node_degree['node degree'].min(),
                                                                         node_degree['node degree'].max()))
            print('edge degree: median+-std (min, max)={:}+-{:} ({:}, {:})'.format(edge_degree['edge degree'].median(),
                                                                                   edge_degree['edge degree'].std(),
                                                                         edge_degree['edge degree'].min(),
                                                                         edge_degree['edge degree'].max()))
            pct_pcg = len(g_list)/df.shape[0]*100
            print(f'percentage of protein-coding genes={pct_pcg:.2f}%')

        return df

    def _keep_pcg(self):
        """return protein coding gene symbol"""
        df = pd.read_csv(self.ncbi, sep="\t", compression='gzip', engine='python')
        df = df[df['type_of_gene']=='protein-coding']
        return df['Symbol'].values.tolist()


    def _gmt2matrix(self, df):
        """
        return gene by feature
        """
        all_gene_list = []
        invalid_list = ['UNREVIEWED', 'UNKNOWN']

        # retrieve data
        data_dict = {}
        for idx, row in df.iterrows():
            row_list = row.values
            item_str = row_list[0].split('\t')[0] # could be pathway, tft, or go
            gene_list = row_list[0].split('\t')[2:] # gene list corresponding to item
            item_list = item_str.split('_')
            
            if len(set(item_list)&set(invalid_list)) == 0:
                if not item_str in data_dict:
                    if len(gene_list) > 25:
                        data_dict[item_str] = gene_list
                        all_gene_list += gene_list



        # create data matrix
        all_gene_list = sorted(set(all_gene_list))
        all_item_list = sorted(data_dict.keys())
        data_arr = np.zeros(( len(all_item_list), len(all_gene_list) ))
        df = pd.DataFrame(data_arr, columns=all_gene_list, index=all_item_list)
        for item, value_list in data_dict.items():
            df.loc[item, value_list] = 1.0
            if df.loc[item].sum() < 25:
                raise ValueError(f'Error, gene list should > 25, got{len(value_list)}')
        df = df.T # gene by pathway/tft/go
        return df
