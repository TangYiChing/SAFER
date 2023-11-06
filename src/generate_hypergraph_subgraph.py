"""
return hypergraph and subgraph
DOSE: C2, CELL C3



GeneSet Hypergraph include
1. c2cp pathway gene sets
2. c3 tf-targeting genes
3. tissue-specific TFs

GeneSet Subgraph include
1. tsp: tissue-specific expressed genes
2. es: essential genes
3. ef: effective genes (gene effect)
4. grn: cell line-specific targeting genes
5. tfgrn: cell line-specific tfs
6. all: all genes
"""

import argparse
import pathlib

import numpy as np
import pandas as pd
import pickle
import scipy.stats as scistat
import sklearn.preprocessing as skpre
import sklearn.feature_extraction as skfet

import CCLE_DataParser as srcCCLEParser
import MSigDB_DataParser as srcMSigDBParser
import GTEx_DataParser as srcGTExParser
import Create_Hypergraph as srcHyperGraph
import Select_Drugpairs as srcDrugpairs

def load_msigdb(msigdb_file, ncbi_file, keep_all=True):
    """return geneset"""
    msigdb = srcMSigDBParser.ParseMSigDB(root=msigdb_file, ncbi=ncbi_file)
    df = msigdb.get_processed_data() 
    if keep_all == False:
        # filter edges by median+-std
        edge = df.sum(axis=0).to_frame(name='edge size')
        median = edge['edge size'].median()
        std = edge['edge size'].std()
        rowfilter = (edge['edge size'] > np.abs(median-std)) & (edge['edge size'] < np.abs(median+std))
        edge_list = edge[ rowfilter ].index.tolist()
        df = df[edge_list]

    # sanity check: remove rows/cols that are all zeros
    df = df.loc[(df.sum(axis=1) != 0), (df.sum(axis=0) != 0)]
    df = df.fillna(0)
    n_single_node = (df.sum(axis=1)==0).sum() # single node: no connected edge at all
    n_single_edge = (df.sum(axis=0)==0).sum() # single edge: no connected node at all
    if n_single_node > 0:
        print(f'WARNNING: GeneSet hypergraph contains {n_single_node} single node that no connected edge at all')
    if n_single_edge > 0:
        print(f'WARNNING: GeneSet hypergraph contains {n_single_edge} single edge that no connected nofr at all')
    return df # gene by pathway/tft/tf

def load_ccle(ccle_file, model_file, omics_str, transform_method=None):
    """return omics"""
    ccle = srcCCLEParser.ParseCCLE(root=ccle_file, omics=omics_str, model=model_file, transformed=transform_method)
    df, model_df = ccle.get_processed_data()
    return df # gene by ACH_ID

def load_gtex(gtex_file):
    """load tissue-specific expressed genes"""
    gtex = srcGTExParser.ParseGTEx(root=gtex_file)
    s_df = gtex.get_s_protein() # gene symbol by gtex tissues, cell values are probability of tissue specificity
    s_arr = s_df.values
    s_arr[s_arr > 0.0 ] = 1 # threshold pass 0.0 considered as tsps
    s_arr[s_arr < 1] = 0
    s_df = pd.DataFrame(s_arr, index=s_df.index, columns=s_df.columns)

    # map GTEX tissues to CCLE tissues
    # note: soft tissue and bone in CCLE are not found in GTEX
    ccle_gtex_tissue_map = {'lung': ['Lung'], 'ovary': ['Ovary'], 'breast':['Breast - Mammary Tissue'],
                            'prostate':['Prostate'], 'skin':['Skin - Not Sun Exposed (Suprapubic)', 'Skin - Sun Exposed (Lower leg)'],
                            'large_intestine':['Colon - Sigmoid', 'Colon - Transverse'],
                            'brain':['Brain - Amygdala', 'Brain - Anterior cingulate cortex (BA24)', 'Brain - Caudate (basal ganglia)',
                            'Brain - Cerebellar Hemisphere', 'Brain - Cerebellum', 'Brain - Cortex', 'Brain - Frontal Cortex (BA9)',
                            'Brain - Hippocampus', 'Brain - Hypothalamus', 'Brain - Nucleus accumbens (basal ganglia)',
                            'Brain - Spinal cord (cervical c-1)', 'Brain - Substantia nigra'], 'haematopoietic_and_lymphoid':['Whole Blood'],
                           'kidney': ['Kidney - Cortex', 'Kidney - Medulla'], 'liver':['Liver'], 'pancrease':['Pancreas'], 'stomach':['Stomach']}
    # collect s_protein for ccle cell lines
    df_list = []
    for tissue, tissue_list in ccle_gtex_tissue_map.items():
        if len(tissue_list) > 1:
            df = s_df[tissue_list].mean(axis=1).to_frame(name=tissue)
        else:
            df = s_df[tissue_list]
        df.columns = [tissue]
        df_list.append(df)
    tsp_df = pd.concat(df_list, axis=1) # gene symbol by ccle tissue
    tsp_arr = tsp_df.values
    tsp_arr[tsp_arr > 0.0 ] = 1 # replace prob>0 with 1
    tsp_df = pd.DataFrame(tsp_arr, index=tsp_df.index, columns=tsp_df.columns)
    return tsp_df # gene by ccle tissue

def generate_Hstats(H, label):
    """display summary stats"""
    # obtain stats
    n_nodes = H.shape[0]
    n_edges = H.shape[1]
    df = H.sum(axis=0).to_frame('edge size')
    median = df['edge size'].median()
    std = df['edge size'].std()
    q3, q1 = np.percentile(df['edge size'], [75 ,25])
    iqr = q3 - q1 # interquartile range
    # create reports
    stat_df = pd.DataFrame({'Hypergraph':[label],
                            'Num. of nodes':[n_nodes],
                            'Num. of hyperedges':[n_edges],
                            'hyperedge size':["median={:.2f}, std.={:.2f} (IQR={:.2f})".format(median, std, iqr)],
                           })
    return stat_df.T

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def parse_parameter():
    parser = argparse.ArgumentParser(description="Return hypergraphs and subgraphs")
    parser.add_argument("-r", "--root",
                        required = True,
                        help = "path to required file folder, e.g., ../database/")
    parser.add_argument("-processed","--processed_path",
                        required = True,
                        help = "path to processed data files from process_raw_data.py, e.g., ./data/processed_data")
    parser.add_argument("-study", "--study",
                        default = 'all',
                        type = str,
                        help = "string representing study name")
    parser.add_argument("-debug", "--DEBUG",
                        action = "store_true",
                        help = "display pring message if enabled by -debug")
    parser.add_argument("-tsp", "--select_tsp",
                        action = "store_true",
                        help = "subsetting summary data to tissue-specific drug pairs if enabled by -tsp, otherwise will use all drug pairs")
    parser.add_argument("-s", "--chem",
                        default = 'smile',
                        help = "chemical structure representation [smile, selfies]. default: smile")
    parser.add_argument("-k", "--kmer",
                        default = 3,
                        type = int,
                        help = "integer representing kmer. defualt=3")
    parser.add_argument("-sg", "--structure_gene",
                        action = "store_true", 
                        help = "add target hypergraph to chemical hypergraph if enabled, otherwise will use chemical structure only")
    parser.add_argument("-m", "--msigdb",
                        default = 'c2-c3-tf',
                        help = "msigdb gene set [c2, c3, c2-c3, c2-c3-tf]. default: c2-c3-tf")
    parser.add_argument("-c", "--ccle",
                        default = 'exp',
                        help = "ccle omics [exp, mut, cnv]. default: exp")
    parser.add_argument("-u", "--use_gene",
                        default = 'all',
                        help = "inject omics readouts [es, ef, tsp, tfgrn, grn, all] to specific genes. default: all")
    parser.add_argument("-dose", "--dose_onehot",
                       action = "store_true", 
                       help = "create one-hot vector for dosing. default: False")
    parser.add_argument("-clinical", "--clinical_onehot",
                        action = "store_true",
                        help = "create one-hot vector for clinical info. default: False")
    parser.add_argument("-fout", "--fout_path",
                        required = True,
                        help = "folder path to store processed data")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_parameter()

    # check required files
    data_dict = {'ccle_exp':f'{args.root}/DepMap/OmicsExpressionProteinCodingGenesTPMLogp1.csv',
                 'ccle_ef':f'{args.root}/DepMap/CRISPRGeneEffect.csv',
                 'ccle_bef':f'{args.root}/DepMap/CRISPRGeneEffect.csv',
                 'ccle_es':f'{args.root}/DepMap/CRISPRGeneDependency.csv',
                 'ccle_bes':f'{args.root}/DepMap/CRISPRGeneDependency.csv',
                 'ccle_mut':f'{args.root}/DepMap/OmicsSomaticMutations.csv',
                 'ccle_cnv':f'{args.root}/DepMap/OmicsCNGene.csv',
                 'ccle_grn':f'{args.root}/GRAND/GRAND.CCLE.Sample-Specific.meanDifferentialTargetingGene.csv',
                 'ccle_bgrn':f'{args.root}/GRAND/GRAND.CCLE.Sample-Specific.meanDifferentialTargetingGene.csv',
                 'ccle_tfgrn':f'{args.root}/GRAND/GRAND.CCLE.Sample-Specific.meanDifferentialTargetingTF.csv',
                 'ccle_btfgrn':f'{args.root}/GRAND/GRAND.CCLE.Sample-Specific.meanDifferentialTargetingTF.csv',
                 'ccle_model':f'{args.root}/DepMap/Model.csv',
                 'msigdb_c2cp':f'{args.root}/MSigDB/c2.cp.v2022.1.Hs.symbols.gmt',
                 'msigdb_c2cgp':f'{args.root}/MSigDB/c2.cgp.v2022.1.Hs.symbols.gmt',
                 'msigdb_c3':f'{args.root}/MSigDB/c3.tft.gtrd.v2022.1.Hs.symbols.gmt',
                 'msigdb_c8':f'{args.root}/MSigDB/c8.all.v2022.1.Hs.symbols.gmt',
                 'tissue_tf':f'{args.root}/GTEx/Tissue-Specific.TF.csv', # tissue-specific variation, ref:DOI: 10.3390/genes13050929
                 'ncbi':f'{args.root}/NCBI/Homo_sapiens.gene_info.gz',
                 'gtex':f'{args.root}/GTEx/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz'}



    # load drugcomb data
    print(f'loading drugcomb summary data')
    db_summary_df = pd.read_pickle(args.processed_path+'drugcomb_processed_summary.pkl')
    db_drug_df = pd.read_pickle(args.processed_path+'drugcomb_processed_drug.pkl')
    db_cell_df = pd.read_pickle(args.processed_path+'drugcomb_processed_cell.pkl')
    db_target_df = pd.read_pickle(args.processed_path+'drugcomb_processed_target.pkl')
    print(db_summary_df.columns)
    if args.clinical_onehot == True:
        print(f'loading cell lines clinical info')
        ccle_model_df = pd.read_csv(data_dict['ccle_model'], header=0, usecols=['ModelID', 'Age', 'Sex'])
        ccle_model_df.columns = ['depmap_id', 'Age', 'Sex']
        db_model_df = db_cell_df[['depmap_id', 'cid']].merge(ccle_model_df, on='depmap_id', how='left')

        # create age group
        bins = [1, 14, 46, 62, 101] # [1,14), [14,46), [46,62), right most not included
        age_groups = ['<15', '15-47', '48-63', '>64']
        db_model_df['age_groups'] = pd.cut(db_model_df['Age'], bins=bins, labels=age_groups, right=False).cat.add_categories('Unknown').fillna('Unknown')
        print(db_model_df['age_groups'].value_counts())
        print(db_model_df['Sex'].value_counts())
        print(db_model_df.head())
    
    # subsetting to include specific study datasets
    s_list = ['ONEIL', 'ALMANAC', 'FORCINA', 'NCATS_ES(NAMPT+PARP)',
              'NCATS_2D_3D', 'NCATS_HL', 'YOHE', 'FLOBAK', 'FRIEDMAN',
              'DYALL', 'FALLAHI-SICHANI', 'ASTRAZENECA', None]
    if args.study != 'all':
        if args.study not in s_list:
            raise ValueError(f'Error, {args.study} not in {s_list}!!!')
        else:
            db_summary_df = db_summary_df[db_summary_df['study_name']==args.study]

    cid_list = sorted(db_summary_df['cid'].unique())
    idx_dict = dict(zip(db_cell_df['depmap_id'], db_cell_df['cid'])) # depmap_id:cid mapping
    

    # load ccle omics data
    #print(f'loading ccle processed data')
    if args.ccle in ['exp', 'cnv']:
        scale_method = None #'MinMax'
    else:
        scale_method = None
    ccle_str = 'ccle_'+args.ccle
    ccle = srcCCLEParser.ParseCCLE(root=data_dict[ccle_str], omics=args.ccle, model=data_dict['ccle_model'],
                                   transformed=scale_method, debug=args.DEBUG)
    ccle_df, ccle_model_df = ccle.get_processed_data()
    ccle_df.rename(columns=idx_dict, inplace=True) # replace depmap_id with cid
    ccle_model_df.loc[:, 'cid'] = ccle_model_df.loc[:, 'depmap_id'].replace(to_replace=idx_dict) # add cid
    cid_tcga_dict = dict(zip(ccle_model_df['cid'], ccle_model_df['tcga abbrev'])) #cid:tcga abbrev mapping
    cid_tissue_dict = dict(zip(db_summary_df['cid'], db_summary_df['tissue_name'])) #cid:tissue_name mapping
    db_summary_df['tcga abbrev'] = db_summary_df['cid'].replace(to_replace=cid_tcga_dict)
    # keep cid only and exclude ach_id
    cid_cols = [col for col in ccle_df.columns if col.startswith('cid')]
    ccle_df = ccle_df[cid_cols]

    # stats
    did_list = [set(db_target_df.columns), set(db_summary_df['did_row']), set(db_summary_df['did_col'])]
    did_list = sorted( set.intersection(*did_list) )
    cid_list = sorted(set(db_summary_df['cid']) & set(ccle_df.columns))
    db_summary_df = db_summary_df[db_summary_df['cid'].isin(cid_list)]
    db_summary_df = db_summary_df[(db_summary_df['did_row'].isin(did_list))&(db_summary_df['did_col'].isin(did_list))]
    db_target_df = db_target_df[did_list]
    ccle_df = ccle_df[cid_list]
    ccle_model_df = ccle_model_df[ccle_model_df['cid'].isin(cid_list)]

    print(f'generating qualified drug combination data for training')
    if args.select_tsp:
        print(f'    use drug pairs with high variable synergy')
        pair = srcDrugpairs.Drugpairs(root=args.processed_path+'drugcomb_processed_summary.pkl',
                            n_tissue=3, n_cell=5, dose_region='all', debug=args.DEBUG)
        pair_df = pair.get_processed_data() # tissue-specific drug pairs summary data
        icombo = sorted(set(pair_df['icombo'])&set(db_summary_df['icombo']))
        pair_df = db_summary_df[db_summary_df['icombo'].isin(icombo)]
        pair_df.to_pickle(f'{args.fout_path}/drugcomb_processed_data.pkl')
    else:
        print(f'    use all drug pairs')
        db_summary_df.to_pickle(f'{args.fout_path}/drugcomb_processed_data.pkl')
    ccle_model_df.to_pickle(f'{args.fout_path}/ccle_model.pkl')
    print(f'    {args.fout_path}/drugcomb_processed_data.pkl')
    print(f'    {args.fout_path}/ccle_model.pkl')

    # collect stats of hypergraphs
    stat_df_list = []
    print(ccle_df.head())
    ##################################################################
    # transform dose concentration
    ##################################################################
    eps = 1e-6
    min_r = abs( np.min(np.log10(db_summary_df['conc_r']+eps)) )
    min_c = abs( np.min(np.log10(db_summary_df['conc_c']+eps)) )
    db_summary_df['converted conc_r'] = np.log10(db_summary_df['conc_r']+eps) + min_r
    db_summary_df['converted conc_c'] = np.log10(db_summary_df['conc_c']+eps) + min_c
    if (db_summary_df['converted conc_r']<0).sum() > 0: 
        raise ValueError(f'Error, conc_r data contains negative values after log conversion!!!')
    if (db_summary_df['converted conc_c']<0).sum() > 0:
        raise ValueError(f'Error, conc_c data contains negative values after log conversion!!!')
    print(db_summary_df[['icombo', 'conc_r', 'converted conc_r', 'conc_c', 'converted conc_c']])

    ##################################################################
    # map drug gene name to ncbi gene symbol
    ##################################################################
    ncbi_df = pd.read_csv(data_dict['ncbi'], sep="\t", compression='gzip', engine='python')
    remain_list = sorted( set(db_target_df.index) - set(ncbi_df['Symbol']) )
    remain_list = [g.replace(' ', '') for g in remain_list if ' ' in g]
    remain_list = [g.replace('Candi', '') for g in remain_list if 'Candi' in g]
    found_symb_dict = {}
    for idx, rows in ncbi_df.iterrows():
        symb_syn_dict = {}
        symbol = rows['Symbol']
        syn_list = rows['Synonyms'].replace('|', ',').split(',')
        syn_list += [symbol]
        symb_syn_dict[symbol] = syn_list
        found_list = sorted(set(syn_list) & set(remain_list))
        if len(found_list) > 0:
            for found in found_list:
                found_symb_dict[found] = symbol
    db_target_df.rename(index=found_symb_dict, inplace=True)
    db_target_df = db_target_df[~db_target_df.index.duplicated(keep='first')]
    hp_list = sorted( set(db_target_df.index) & set(ncbi_df['Symbol']) ) # keep human genes
    db_target_df = db_target_df.loc[hp_list]

    n = len(set(db_target_df.index) & set(ccle_df.index))
    print(f'    #{n} genes overlapping between drug target and omics')



    ##################################################################
    # generate gene set hypergraph
    # 1. c2cp pathway gene sets
    # 2. tissue-specific TFs
    # 3. c3 grtd tf-targeting gene sets
    ##################################################################
    c2_df = load_msigdb(msigdb_file=data_dict['msigdb_c2cp'], ncbi_file=data_dict['ncbi'], keep_all=False) # gene symbol by msigdb pathways
    c3_df = load_msigdb(msigdb_file=data_dict['msigdb_c3'], ncbi_file=data_dict['ncbi'], keep_all=False) # gene symbol by msigdb tf
    tf_df = pd.read_csv(data_dict['tissue_tf'], header=0, index_col=0) # gene symbol by human tissue
    

    #################################################################
    # Load other cell line data
    # 1. essential genes
    # 2. essential genes (effect)
    # 3. cell line-specific TFs
    # 4. cell line-specific TF-targeting genes
    # 5. tissue-specific expressed genes
    #################################################################
    #es_df = load_ccle(ccle_file=data_dict['ccle_bes'], model_file=data_dict['ccle_model'], omics_str='bes', transform_method=None)
    #ef_df = load_ccle(ccle_file=data_dict['ccle_bef'], model_file=data_dict['ccle_model'], omics_str='bef', transform_method=None)
    #tfgrn_df = load_ccle(ccle_file=data_dict['ccle_btfgrn'], model_file=data_dict['ccle_model'], omics_str='btfgrn', transform_method=None)
    #grn_df = load_ccle(ccle_file=data_dict['ccle_bgrn'], model_file=data_dict['ccle_model'], omics_str='bgrn', transform_method=None)
    #tsp_df = load_gtex(gtex_file=data_dict['gtex'])
    #target_df = db_target_df.copy()

    # map depmap_id to cid
    #es_df.rename(columns=idx_dict, inplace=True)
    #ef_df.rename(columns=idx_dict, inplace=True)
    #tfgrn_df.rename(columns=idx_dict, inplace=True)
    #grn_df.rename(columns=idx_dict, inplace=True)


    if args.DEBUG == True:
        print(f'GeneSet Hypergraph: C2')
        print(f'    #genes={c2_df.shape[0]}')
        print(f'    average genes per pathwas={c2_df.sum(axis=0).mean()}')
        print(f'GeneSet Hypergraph: C3')
        print(f'    #genes={c3_df.shape[0]}')
        print(f'    average genes per pathwas={c3_df.sum(axis=0).mean()}')
        print(f'GeneSet Hypergraph: Tissue-Specific TFs')
        print(f'    #genes={tf_df.shape[0]}')
        print(f'    average genes per pathwas={tf_df.sum(axis=0).mean()}')
        
        print(f'GeneSet: Essential genes')
        print(f'    average essential genes per ccle cell line={es_df.sum(axis=0).mean()}')
        print(f'GeneSet: Essential genes (effect)')
        print(f'    average essential genes (normalized) per ccle cell line={ef_df.sum(axis=0).mean()}')
        print(f'GeneSet: cell line-specific TFs')
        print(f'    average cell line-specific TFs per ccle cell line={tfgrn_df.sum(axis=0).mean()}')
        print(f'GeneSet: cell line-specific TF-targeting genes')
        print(f'    average cell line-specific TF-targeting genes per ccle cell line={grn_df.sum(axis=0).mean()}')
        print(f'GeneSet: tissue-specific expressed genes')
        print(f'    average tissue-specific expressed genes per ccle tissue={tsp_df.sum(axis=0).mean()}')
        print(f'GeneSet: drug genes')
        print(f'    average drug-associated genes per drug={target_df.sum(axis=0).mean()}')


    #####################################################################
    # define geneset
    # 
    #
    #####################################################################
    if args.msigdb == 'c2':
        msigdb_df = c2_df
    elif args.msigdb == 'c3':
        msigdb_df = c3_df
    elif args.msigdb == 'tf':
        msigdb_df = tf_df
    elif args.msigdb == 'c2-c3':
        row_list = sorted(set(c2_df.index) | set(c3_df.index))
        col_list = sorted(set(c2_df.columns) | set(c3_df.columns))
        msigdb_df = pd.DataFrame(np.zeros((len(row_list),len(col_list))), index=row_list, columns=col_list)
        msigdb_df.loc[c2_df.index, c2_df.columns] = c2_df
        msigdb_df.loc[c3_df.index, c3_df.columns] = c3_df
        
    elif args.msigdb == 'c2-c3-tf':
        row_list = set(c2_df.index).union(set(c3_df.index),set(tf_df.index))
        col_list = set(c2_df.columns).union(set(c3_df.columns),set(tf_df.columns))
        msigdb_df = pd.DataFrame(np.zeros((len(row_list),len(col_list))), index=row_list, columns=col_list)
        msigdb_df.loc[c2_df.index, c2_df.columns] = c2_df
        msigdb_df.loc[c3_df.index, c3_df.columns] = c3_df
        msigdb_df.loc[tf_df.index, tf_df.columns] = tf_df
    else:
        raise ValueError(f'{args.msigdb} not supported!!!')

    ## transform drug gene by did to msigdb by did
    g_list = sorted(set(msigdb_df.index) & set(db_target_df.index))
    m = msigdb_df.loc[g_list] # gene by msigdb
    d = db_target_df.loc[g_list] # gene by did
    msigdb_drug_df = pd.DataFrame(np.zeros((m.shape[1], d.shape[1])), index=m.columns, columns=d.columns) # msigdb by did
    for msigdb in m.columns:
        for did in d.columns:
            s = (d[did] * m[msigdb]).sum()
            if s > 0:
                msigdb_drug_df.loc[msigdb, did] = s
    print(f' summary of #msigdb per drug: {msigdb_drug_df.mean().describe()}')


    ## transform cell line gene by cid to msigdb by cid
    if args.msigdb == 'c2':
        if args.ccle == 'exp':
            msigdb_cell_df = pd.read_pickle(f'{args.root}/ssGSEA/ccle_exp.c2.cp.v2022.1.Hs.symbols.pkl')
        elif args.ccle == 'es':
            msigdb_cell_df = pd.read_pickle(f'{args.root}/ssGSEA/ccle_es.c2.cp.v2022.1.Hs.symbols.pkl')
        elif args.ccle == 'tfgrn':
            msigdb_cell_df = pd.read_pickle(f'{args.root}/ssGSEA/ccle_tfgrn.c2.cp.v2022.1.Hs.symbols.pkl')
        else:
            raise ValueError(f'{args.ccle} for ssGSEA not supported!')
    elif args.msigdb == 'c3':
        if args.ccle == 'exp':
            msigdb_cell_df = pd.read_pickle(f'{args.root}/ssGSEA/ccle_exp.c3.tft.gtrd.v2022.1.Hs.symbols.pkl')
        elif args.ccle == 'es':
            msigdb_cell_df = pd.read_pickle(f'{args.root}/ssGSEA/ccle_es.c3.tft.gtrd.v2022.1.Hs.symbols.pkl')
        elif args.ccle == 'tfgrn':
            msigdb_cell_df = pd.read_pickle(f'{args.root}/ssGSEA/ccle_tfgrn.c3.tft.gtrd.v2022.1.Hs.symbols.pkl')
        else:
            raise ValueError(f'{args.ccle} for ssGSEA not supported!')
    else:
        if args.ccle == 'exp':
            c2 = pd.read_pickle(f'{args.root}/ssGSEA/ccle_exp.c2.cp.v2022.1.Hs.symbols.pkl')
            c3 = pd.read_pickle(f'{args.root}/ssGSEA/ccle_exp.c3.tft.gtrd.v2022.1.Hs.symbols.pkl')
        elif args.ccle == 'es':
            c2 = pd.read_pickle(f'{args.root}/ssGSEA/ccle_es.c2.cp.v2022.1.Hs.symbols.pkl')
            c3 = pd.read_pickle(f'{args.root}/ssGSEA/ccle_es.c3.tft.gtrd.v2022.1.Hs.symbols.pkl')
        elif args.ccle == 'tfgrn':
            c2 = pd.read_pickle(f'{args.root}/ssGSEA/ccle_tfgrn.c2.cp.v2022.1.Hs.symbols.pkl')
            c3 = pd.read_pickle(f'{args.root}/ssGSEA/ccle_tfgrn.c3.tft.gtrd.v2022.1.Hs.symbols.pkl')
        else:
            raise ValueError(f'{args.ccle} for ssGSEA not supported!')
        col_list = sorted( set(c2.columns)&set(c3.columns) )
        c2 = c2[col_list]
        c3 = c3[col_list]
        msigdb_cell_df = pd.concat([c2,c3], axis=0)
    col_list = [idx_dict[col] if col in idx_dict else col for col in msigdb_cell_df.columns] 
    msigdb_cell_df.columns = col_list
    msigdb_cell_df = msigdb_cell_df[ sorted(set(col_list)&set(cid_list)) ]
    print(msigdb_cell_df)
    print(f' summary of #msigdb per cell: {msigdb_cell_df.mean().describe()}')
    #####################################################################
    # generate hypergraphs
    #####################################################################
    # 1. generate dose hypergraph
    print(f'creating DOSE hypergraph')
    g_list = sorted(set(ccle_df.index) & set(msigdb_df.index))
    msigdb_df = msigdb_df.loc[g_list]
    ccle_df = ccle_df.loc[g_list]
    print(f'    #genes={len(g_list)} overlapping between ccle_df and msigdb:{args.msigdb}')
    dose_hyp_df = msigdb_df.T # transform: msigdb by gene
    gene = srcHyperGraph.GeneHypergraph(incidence_matrix=dose_hyp_df, transform=False,
                                        graph_regularization=True,debug=args.DEBUG)
    idx_list, col_list, H, G = gene.construct_H_G()
    gene_dict = {'hypergraph': (H, G),
                 'index':idx_list,
                 'columns':col_list}
    stat_df_list.append( generate_Hstats(dose_hyp_df, 'Dose') )

    # 2. generate chemical hypergraph
    print(f'creating Chemical hypergraph')
    db_drug_df = db_drug_df[db_drug_df['did'].isin(did_list)] # did, smiles
    smile = srcHyperGraph.SmileHypergraph(smile_df=db_drug_df, chem=args.chem, kmer=args.kmer, transform=False,
                                          graph_regularization=True, debug=False)
    idx_list, col_list, H, G = smile.construct_H_G()
    drug_dict = {'hypergraph': (H, G),
                 'index':idx_list,
                 'columns':col_list}
    kmer_hyp_df = pd.DataFrame(H, index=idx_list, columns=col_list)
    stat_df_list.append( generate_Hstats(kmer_hyp_df, 'Structure kmer={:}'.format(args.kmer)) )

    if args.structure_gene == True:
        print(f'    adding drug gene hypergraph')
        # drug gene
        did_list = sorted(set(kmer_hyp_df.columns)&set(msigdb_drug_df.columns))
        kmer_hyp_df = kmer_hyp_df[did_list] # kmer by did
        target_hyp_df = msigdb_drug_df[did_list] # msigdb by did

        # merge structure and drug gene hypergraphs
        hyp_df = pd.concat([kmer_hyp_df, target_hyp_df], axis=0)
        drug = srcHyperGraph.GeneHypergraph(incidence_matrix=hyp_df, transform=False,
                                            graph_regularization=True,debug=args.DEBUG)
        idx_list, col_list, H, G = drug.construct_H_G()
        drug_dict = {'hypergraph': (H, G),
                     'index':idx_list,
                     'columns':col_list}
        stat_df_list.append( generate_Hstats(hyp_df, 'Structure-Gene') )
        chemical_hyp_df = hyp_df
    else:
        chemical_hyp_df = kmer_hyp_df

    ##################################################################
    # save to files
    ##################################################################
    with open(f'{args.fout_path}/dose_hypergraph.pkl', 'wb') as f:
        pickle.dump(gene_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{args.fout_path}/chemical_hypergraph.pkl', 'wb') as f:
        pickle.dump(drug_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'generating hypergraphs')
    print(f'    {args.fout_path}/dose_hypergraph.pkl')
    print(f'    {args.fout_path}/chemical_hypergraph.pkl')


    #####################################################################
    # generate subgraphs
    #####################################################################
    print(f'creating Dose hypergraph')
    #geneset_dict = {'es':es_df, 'ef':ef_df, 'tfgrn':tfgrn_df, 'grn':grn_df, 'tsp':tsp_df, 'drug':target_df}
    
    # 1. prepare omics readouts
    omics_df = pd.DataFrame(np.zeros( (dose_hyp_df.shape[0], msigdb_cell_df.shape[1]) ),
                                index=dose_hyp_df.index, columns=msigdb_cell_df.columns )
    g_list = sorted(set(dose_hyp_df.index)&set(msigdb_cell_df.index))
    omics_df.loc[msigdb_cell_df.loc[g_list].index, msigdb_cell_df.loc[g_list].columns] = msigdb_cell_df.loc[g_list] # msigdb by cid

    # 2. prepare drug genes
    #drug_gene_df = pd.DataFrame(np.zeros( (dose_hyp_df.shape[0], msigdb_drug_df.shape[1]) ),
    #                            index=dose_hyp_df.index, columns=msigdb_drug_df.columns )
    #g_list = sorted(set(dose_hyp_df.index)&set(migdb_drug_df.index))
    #drug_gene_df.loc[msigdb_drug_df.loc[g_list].index, msigdb_drug_df.loc[g_list].columns] = msigdb_drug_df.loc[g_list] # msigdb by did

    # 3. prepare other genesets
    #for key, df in geneset_dict.items():
    #    gene_df = pd.DataFrame(np.zeros( (dose_hyp_df.shape[0], df.shape[1]) ),
    #                          index=dose_hyp_df.index, columns=df.columns )
    #    g_list = sorted(set(dose_hyp_df.index)&set(df.index))
    #    gene_df.loc[df.loc[g_list].index, df.loc[g_list].columns] = df.loc[g_list] # gene by cid
    #    geneset_dict[key] = gene_df # same set of gene as hypergraph
            

    # create subgraph
    icombo_list = sorted(db_summary_df['icombo'].unique())
    dose_subg_df = pd.DataFrame(np.zeros((len(dose_hyp_df.index), len(icombo_list))),
                                index=dose_hyp_df.index, columns=icombo_list)
    chemical_subg_df = pd.DataFrame(np.zeros((len(chemical_hyp_df.index), len(icombo_list))),
                                index=chemical_hyp_df.index, columns=icombo_list)

    # create dosing onehot vector
    dosing_list = sorted( set(db_summary_df['conc_r'].unique()) | set(db_summary_df['conc_c'].unique()) )
    rc_list = []

    # create clinical onehot vector
    age_list = sorted(db_model_df['age_groups'].unique())
    sex_list = sorted(db_model_df['Sex'].unique())
    tissue_list = sorted(db_summary_df['tissue_name'].unique())
    clinical_grps = db_model_df.groupby('cid')
    clinical_list = []

    # loop through icombo subject to create subgraph
    grps = db_summary_df.groupby('icombo')
    for icombo in icombo_list:
        did_r = icombo.split('=')[0]
        did_c = icombo.split('=')[1]
        cid = icombo.split('=')[2]
        conc_r = grps.get_group(icombo)['converted conc_r'].values[0]
        conc_c = grps.get_group(icombo)['converted conc_c'].values[0]
        dosage = conc_r + conc_c
        dosing_r = icombo.split('=')[3].split('r')[1]
        dosing_c = icombo.split('=')[4].split('c')[1]
        tissue = cid_tissue_dict[cid]
        omics_dosage = omics_df[cid] * dosage
            
        ############
        #dose module
        ############
        # 2. injecting cell omics readouts to all genes
        #if args.use_gene == 'tsp': # injecting cell omics readouts into tissue-specific expressed genes
        #    if tissue in geneset_dict['tsp'].columns:
        #        a = geneset_dict['tsp'][tissue] * omics_dosage
        #    else:
        #        a = omics_dosage
        #elif args.use_gene == 'all':
        #    a = omics_dosage
        #else:
        #    if cid in geneset_dict[args.use_gene].columns: # can be [es, ef, tfgrn, grn, drug]
        #        a = geneset_dict[args.use_gene][cid] * omics_dosage
        #    else:
        #        a = omics_dosage

        # update subject values to subgraph
        dose_subg_df.loc[:, icombo] =  omics_dosage

        ################
        #chemical module
        ################
        # drug structure
        chemical_subg_df.loc[chemical_hyp_df.index, icombo] = chemical_hyp_df.loc[:, did_r] * float(conc_r) + chemical_hyp_df.loc[:, did_c] * float(conc_c)
        if args.structure_gene == True:
            # drug gene
            chemical_subg_df.loc[target_hyp_df.index, icombo] = target_hyp_df.loc[:, did_r] * float(conc_r) + target_hyp_df.loc[:, did_c] * float(conc_c)

        ###############
        #dosing onehot
        ###############
        if args.dose_onehot == True:
            r_idx = dosing_list.index( float(dosing_r) )
            c_idx = dosing_list.index( float(dosing_c) )
            r_arr = np.zeros((1,len(dosing_list)))
            c_arr = np.zeros((1,len(dosing_list)))
            r_arr[0, r_idx] = 1
            c_arr[0, c_idx] = 1
            rc_arr = np.concatenate([r_arr, c_arr], axis=1)
            rc_list.append(rc_arr)
            if rc_arr.sum() != 2:
                raise ValueError(f'icombo={icombo} missing dosing r or dosing c!!!')

        ###############
        #clinical onehot
        ###############
        if args.clinical_onehot == True:
            cid_info = clinical_grps.get_group(cid)
            age_idx = age_list.index( cid_info['age_groups'].values[0] )
            sex_idx = sex_list.index( cid_info['Sex'].values[0] )
            tissue_idx = tissue_list.index( tissue )
            age_arr = np.zeros((1, len(age_list)))
            sex_arr = np.zeros((1, len(sex_list)))
            tissue_arr = np.zeros((1, len(tissue_list)))
            age_arr[0, age_idx] = 1
            sex_arr[0, sex_idx] = 1
            tissue_arr[0, tissue_idx] = 1
            clinical_arr = np.hstack([age_arr, sex_arr, tissue_arr])
            clinical_list.append(clinical_arr)
            if clinical_arr.sum() != 3:
                raise ValueError(f'icombo={icombo} missing one of clinical info: age, sex, tissue!!!')

    # concatenate dosing onehot vectors
    if args.dose_onehot == True:
        dosing_df = pd.DataFrame(np.concatenate(rc_list, axis=0), index=icombo_list)
        if args.clinical_onehot == True:
            print(f'appending clinical onehot to dosing onehot')
            clinical_df = pd.DataFrame(np.concatenate(clinical_list, axis=0), index=icombo_list)
            dosing_df = pd.concat([dosing_df, clinical_df], axis=1)
            dosing_df.to_pickle(f'{args.fout_path}/dosing_onehot.pkl')
            print(f'    distribution of dosing onehot:\n{dosing_df.mean(axis=1).describe()}')
            print(f'generating dosing onehot')
            print(f'    {args.fout_path}/dosing_onehot.pkl')
        else:
            dosing_df.to_pickle(f'{args.fout_path}/dosing_onehot.pkl')
            print(f'    distribution of dosing onehot:\n{dosing_df.mean(axis=1).describe()}')
            print(f'generating dosing onehot')
            print(f'    {args.fout_path}/dosing_onehot.pkl')
    else:
        if args.clinical_onehot == True:
            print(f'save clinical onehot without dosing onehot')
            clinical_df = pd.DataFrame(np.concatenate(clinical_list, axis=0), index=icombo_list)
            clinical_df.to_pickle(f'{args.fout_path}/dosing_onehot.pkl')
            print(f'    distribution of dosing onehot:\n{clinical_df.mean(axis=1).describe()}')
            print(f'generating dosing onehot')
            print(f'    {args.fout_path}/dosing_onehot.pkl')

    # sanity check subject values
    if dose_subg_df.sum().sum() == 0:
        raise ValueError(f'Error, dose subgraph is empty!!!')
    else:
        print(f'    distribution of drug combination subjuect values (DOSE):\n{dose_subg_df.mean(axis=0).describe()}')

    if chemical_subg_df.sum().sum() == 0:
        raise ValueError(f'Error, chemical subgraph is empty!!!')
    else:
        print(f'    distribution of drug combination subjuect values (CHEM):\n{chemical_subg_df.mean(axis=0).describe()}')

    ##################################################################
    # save to files
    ##################################################################
    dose_subg_df.to_pickle(f'{args.fout_path}/dose_subgraph.pkl')
    chemical_subg_df.to_pickle(f'{args.fout_path}/chemical_subgraph.pkl')
    print(f'generating subgraph')
    print(f'    {args.fout_path}/dose_subgraph.pkl')
    print(f'    {args.fout_path}/chemical_subgraph.pkl')

    #####################################################################
    ## Hypergraph Stats
    ## Data Stats
    #####################################################################
    # finalize summary reports
    stat_df = pd.concat(stat_df_list, axis=1) # hypergraphs

    # stats of training data
    if args.select_tsp == True:
        db_summary_df = pair_df
    did_list = [set(db_summary_df['did_row']), set(db_summary_df['did_col'])]
    did_list = sorted( set.intersection(*did_list) )
    n_did = len(did_list)
    n_cid = len(db_summary_df['cid'].unique())
    n_com = len(db_summary_df.groupby(['did_row', 'did_col', 'cid']).groups.keys())
    n_pair = len(db_summary_df['ipair'].unique())
    n_study_id = len(db_summary_df['study_name'].unique())
    n_tissue = len(db_summary_df['tissue_name'].unique())
    n_ct = len(db_summary_df['tcga abbrev'].unique())
    data_df = pd.DataFrame({'Total data':[db_summary_df.shape[0]],
                            'Num. of drug combinations':[n_com],
                            'Num. of drug pairs':[n_pair],
                            'Num. of cell lines':[n_cid],
                            'Num. of studies':[n_study_id],
                            'Num. of tissues':[n_tissue],
                            'Num. of cancer types':[n_ct]})
    stat_df.to_csv(f'{args.fout_path}/hypergraphs_stats.txt', header=True, index=True, sep="\t")
    data_df.T.to_csv(f'{args.fout_path}/data_stats.txt', header=True, index=True, sep="\t")
    print(f'generating data stats')
    print(f'    {args.fout_path}/hypergraphs_stats.txt')
    print(f'    {args.fout_path}/data_stats.txt')
