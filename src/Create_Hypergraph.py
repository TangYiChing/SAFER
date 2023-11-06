"""
Create Hypergraph (H: incidence matrix)

GeneHypergraph: return Hypergraph
SmileHypergraph: return Hypergraph, subgraph per smile 
"""

import numpy as np
import pandas as pd
import networkx as nx
import pathlib
import pickle
import re
from rdkit import Chem as rdkChem
import sklearn.feature_extraction as skfet
import selfies as sf

class GeneHypergraph:
    def __init__(self, incidence_matrix=None, transform=True, graph_regularization=True, debug=False):

        self.incidence_matrix = incidence_matrix
        self.transform = transform
        self.graph_regularization = graph_regularization
        self.debug = debug

    def _H_stats(self, H):
        """display summary stats"""
        n_nodes = H.shape[0]
        n_edges = H.shape[1]
        df = H.sum(axis=0).to_frame('edge size')
        mean = df['edge size'].mean()
        std = df['edge size'].std()
        minv = df['edge size'].min()
        maxv = df['edge size'].max()
        print(f'n_nodes={n_nodes} | n_edges={n_edges}')
        print(f'edge size: mean, std (min, max)={mean:.2f},{std:.2f}, ({minv:.2f},{maxv:.2f})')

    def construct_H(self):
        """return H"""
        # set H
        H = self.incidence_matrix
        
        #self._H_stats(H)

        # transform
        if self.transform:
            transformer = skfet.text.TfidfTransformer()
            H_arr = transformer.fit_transform(H.values).todense()
            H = pd.DataFrame(H_arr, index=H.index, columns=H.columns)

        if self.debug:
            print(H)
            print(f'H={H.shape} (#genes, #features)')

        # return
        idx_list = H.index.tolist()
        col_list = H.columns.tolist()
        h_arr = H.values
        return idx_list, col_list, h_arr

    def construct_H_G(self):
        """return H, G"""
        idx_list, col_list, h_arr = self.construct_H()

        if self.graph_regularization:
            G = self._generate_G_from_H(h_arr)
            if G.sum() < 0:
                raise ValueError(f'Error, G may contains nan: G.sum={G.sum()}!!!')
        else:
            HT = H.transpose()
            G = np.matmul(H, HT)
        return idx_list, col_list, h_arr, G

    def _generate_G_from_H(self, H, variable_weight=False):
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
            G = np.nan_to_num(G, nan=0.0, posinf=0.0, neginf=0.0)
            return G



class SmileGraph:
    def __init__(self, smile_df=None, debug=False):
        if smile_df is None or len(smile_df) == 0:
            raise Exception(f'Error, smile_df is empty!!!')
        else:
            self.smile_df = smile_df
        self.debug = debug

        self.atom_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                          'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                          'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                          'Pt', 'Hg', 'Pb', 'Unknown'] 
        # blacklist contains smiles that do not have bonds
        self.blacklist = ['[O-2].[O-2].[O-2].[As+3].[As+3]', '[K+].[I-]']
    def construct_graph(self):
        """
        adopted smile_to_graph(smile) from DeepDDs 
        reference: https://github.com/Sinwang404/DeepDDs/blob/master/creat_data_DC.py
        """
        # collect data
        data_dict = {} # dictionary {did: (c_size, features, edge_index)}

        for did in self.smile_df['did'].unique():
            smile = self.smile_df[self.smile_df['did']==did]['smiles'].values[0]
            if ';' in smile:
                smile = smile.split(';')[0]
            mol = rdkChem.MolFromSmiles(smile) # convert to molecular structure

            c_size = mol.GetNumAtoms()
            features = []
            for atom in mol.GetAtoms():
                feature = self.atom_features(atom)
                features.append(feature / sum(feature))
            features = np.sum(features, axis=0)
            
            edges = []
            for bond in mol.GetBonds():
                edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            g = nx.Graph(edges).to_directed()
            edge_index = []
            for e1,e2 in g.edges:
                edge_index.append([e1,e2])
            # append to dict
            data_dict[did] = (c_size, features, edge_index)
        if self.debug:
            print(f'c_size={c_size} |features={features} | edge_index={edge_index}')
        return data_dict

    def _one_of_k_encoding(self, x, allowable_set):
        if x not in allowable_set:
            raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
        return list(map(lambda s: x == s, allowable_set))
    def _one_of_k_encoding_unk(self, x, allowable_set):
        """Maps inputs not in the allowable set to the last element."""
        if x not in allowable_set:
            x = allowable_set[-1]
        return list(map(lambda s: x == s, allowable_set))
    def atom_features(self, atom):
        return np.array(self._one_of_k_encoding_unk(atom.GetSymbol(), self.atom_list) +
                        self._one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                        self._one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                        self._one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                        [atom.GetIsAromatic()])

class SmileHypergraph:
    def __init__(self, smile_df=None, chem='smile', kmer=3, transform=True, graph_regularization=True, debug=False):
        if smile_df is None or len(smile_df) == 0:
            raise Exception(f'Error, smile_df is empty!!!')
        else:
            self.smile_df = smile_df
        self.debug = debug
        self.chem = chem
        self.kmer = kmer
        self.transform = transform
        self.graph_regularization = graph_regularization

    def _H_stats(self, H):
        """display summary stats"""
        n_nodes = H.shape[0]
        n_edges = H.shape[1]
        df = H.sum(axis=0).to_frame('edge size')
        mean = df['edge size'].mean()
        std = df['edge size'].std()
        minv = df['edge size'].min()
        maxv = df['edge size'].max()
        print(f'n_nodes={n_nodes} | n_edges={n_edges}')
        print(f'edge size: mean, std (min, max)={mean:.2f},{std:.2f}, ({minv:.2f},{maxv:.2f})')

    def _split_kmers(self, chem, smile_str, k=3):
        """return list of k-mer given a smile string"""
    
        n = len(smile_str)
        size = n-k+1
        kmer_list = []
        if chem == 'smile':
            for i in range(0, size):
                kmer_list.append( smile_str[i:(i+k)] )
            
        elif chem == 'selfies':
            smile_str = sf.encoder(smile_str)
            smile_str = re.findall('\[.*?\]',smile_str)
            for i in range(0, size):
                kmer_str = ''.join(kmer for kmer in smile_str[i:(i+k)])
                kmer_list.append( kmer_str )        
        else:
            raise ValueError(f'Error, chem should be either SMILE or SELFIES, but got {chem}!!!')
        return kmer_list

    def _collect_nodes(self):
        """return all kmers from all smiles"""
        kmer_dict = {} # did:kmer_list
        for did in self.smile_df['did'].unique():
            smile = self.smile_df[self.smile_df['did']==did]['smiles'].values[0]
            if ';' in smile:
                smile = smile.split(';')[0]
            kmer_dict[did] = self._split_kmers(chem=self.chem, smile_str=smile, k=self.kmer)
        return kmer_dict

    def construct_H(self):
        """return H incidence matrix"""
        kmer_dict = self._collect_nodes()
        node_list = list(sorted({kmer for kmer_list in kmer_dict.values() for kmer in kmer_list}))
        hyperedge_list = list(kmer_dict.keys())

        H = pd.DataFrame( np.zeros( (len(node_list),len(hyperedge_list)) ),
                          index=node_list, columns=hyperedge_list)
        for did, kmer_list in kmer_dict.items():
            H.loc[kmer_list, did] = 1

        #self._H_stats(H)
        # transform
        if self.transform:
            transformer = skfet.text.TfidfTransformer()
            H_arr = transformer.fit_transform(H.values).todense()
            H = pd.DataFrame(H_arr, index=H.index, columns=H.columns)

        if self.debug:
            print(H)
            print(f'H={H.shape} (#kmers, #drugs)')

        # return
        idx_list = H.index.tolist()
        col_list = H.columns.tolist()
        h_arr = H.values
        return idx_list, col_list, h_arr

    def construct_H_G(self):
        """return H, G"""
        idx_list, col_list, h_arr = self.construct_H()

        if self.graph_regularization:
            G = self._generate_G_from_H(h_arr)
            if G.sum() < 0:
                raise ValueError(f'Error, G may contains nan: G.sum={G.sum()}!!!')
        else:
            HT = H.transpose()
            G = np.matmul(H, HT)
        return idx_list, col_list, h_arr, G

    def _generate_G_from_H(self, H, variable_weight=False):
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
            G = np.nan_to_num(G, nan=0.0, posinf=0.0, neginf=0.0)
            return G
