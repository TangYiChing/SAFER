"""
return processed data
1. drugcomb summary data
2. drugcomb drug data
3. drugcomb cell data
4. drug target data
"""

import argparse
import pathlib

import numpy as np
import pandas as pd
import pickle
import scipy.stats as scistat
import sklearn.preprocessing as skpre
import sklearn.feature_extraction as skfet

import DrugComb_DataParser as srcDrugCombParser
import CCLE_DataParser as srcCCLEParser
import MSigDB_DataParser as srcMSigDBParser
import Create_Hypergraph as srcHyperGraph
import STITCH_DataParser as srcSTITCHParser
import DrugBank_DataParser as srcDrugBankParser
import TTD_DataParser as srcTTDParser

def long2wide(target_df, debug=False):
    """ return gene by did"""
    gene_list = list(target_df['symbol'].unique())
    did_list =  [did for did in target_df['did'].unique() if did.startswith('did')]
    df = pd.DataFrame(np.zeros( (len(gene_list), len(did_list)) ),
                      index=gene_list, columns=did_list)
    for j in did_list:
        i = target_df[target_df['did']==j]['symbol'].values.tolist()
        df.loc[i, j] = 1

    if debug:
        print(df)
        print(f'{df.sum()}')
    return df

def parse_parameter():
    parser = argparse.ArgumentParser(description="Return processed datasets")
    parser.add_argument("-r", "--root",
                        required = True,
                        help = "path to required file folder, e.g., ./database/")
    parser.add_argument("-debug", "--DEBUG",
                        default = False,
                        action = "store_true",
                        help = "display pring message if True")
    parser.add_argument("-fout", "--fout_path",
                        required = True,
                        help = "folder path to store processed data")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_parameter()

    # check required files
    data_dict = {'summary':f'{args.root}/DrugComb/summary_v_1_5.csv',
                 'dose':f'{args.root}/DrugComb/dose_response.pkl',
                 'ccle_exp':f'{args.root}/DepMap/OmicsExpressionProteinCodingGenesTPMLogp1.csv',
                 'ccle_ef':f'{args.root}/DepMap/CRISPRGeneEffect.csv',
                 'ccle_es':f'{args.root}/DepMap/CRISPRGeneDependency.csv',
                 'ccle_mut':f'{args.root}/DepMap/OmicsSomaticMutations.csv',
                 'ccle_cnv':f'{args.root}/DepMap/OmicsCNGene.csv',
                 'ccle_model':f'{args.root}/DepMap/Model.csv',
                 'ccle_tfgrn':f'{args.root}/GRAND/GRAND.CCLE.Sample-Specific.meanDifferentialTargetingTF.csv',
                 'ccle_grn':f'{args.root}/GRAND/GRAND.CCLE.Sample-Specific.meanDifferentialTargetingGene.csv',
                 'msigdb_c2cp':f'{args.root}/MSigDB/c2.cp.v2022.1.Hs.symbols.gmt',
                 'msigdb_c2cgp':f'{args.root}/MSigDB/c2.cgp.v2022.1.Hs.symbols.gmt',
                 'msigdb_c3':f'{args.root}/MSigDB/c3.tft.gtrd.v2022.1.Hs.symbols.gmt',
                 'msigdb_c8':f'{args.root}/MSigDB/c8.all.v2022.1.Hs.symbols.gmt',
                 'ncbi':f'{args.root}/NCBI/Homo_sapiens.gene_info.gz',
                 'stitch_target':f'{args.root}/DrugComb/stitch_target.pkl',
                 'ttd_target':f'{args.root}/DrugComb/ttd_target.pkl',
                 'ss':f'{args.root}/DrugComb/drug_selfies.pkl',
                 'cc':f'{args.root}/ChemicalChecker/drug_chemicalchecker.pkl'}

    ########################################################################################
    # Load drugcomb datasets
    # 1. summary data
    # 2. dose response matrix
    ########################################################################################
    if pathlib.Path(args.root+'/drugcomb_combo_dose.pkl').is_file():
        print(f'parsing DrugComb datasets from files')
        combo_df = pd.read_pickle(args.root+'/drugcomb_combo_dose.pkl')
        drug_df = pd.read_pickle(args.root+'/drugcomb_drug.pkl')
        cell_df = pd.read_pickle(args.root+'/drugcomb_cell.pkl')
    else:
        print(f'parsing DrugComb datasets from database')
        drugcomb = srcDrugCombParser.ParseDrugComb(root=data_dict['summary'], debug=args.DEBUG)
        combo_df, drug_df, cell_df = drugcomb.get_processed_data()
        combo_df.to_pickle(args.root+'/drugcomb_combo_dose.pkl')
        drug_df.to_pickle(args.root+'/drugcomb_drug.pkl')
        cell_df.to_pickle(args.root+'/drugcomb_cell.pkl')

    ########################################################################################
    # Check for data availability
    # drugs with SMILE and target
    ########################################################################################
    # keep drug-drug-cell that has SMILE and drug target data
    cell_list = sorted(cell_df['name'].unique())
    drug_list = sorted(drug_df['dname'].unique())
    rowfilter = (combo_df.cell_line_name.isin(cell_list)) & \
                (combo_df.drug_col.isin(drug_list)) &  \
                (combo_df.drug_row.isin(drug_list))
    summary_df = combo_df[rowfilter]
    

    # convert item to index for dname, cell_line_name
    drug_list = sorted( list(set(summary_df['drug_row']).union(set(summary_df['drug_col']))) )
    drug_idx_dict = { drug_list[i]:'did_'+str(i) for i in range(0, len(drug_list)) }

    cell_list = sorted( list(summary_df['cell_line_name'].unique()) )
    cell_idx_dict = { cell_list[i]:'cid_'+str(i) for i in range(0, len(cell_list)) }

    summary_df.loc[:, 'did_row'] = summary_df.loc[:, 'drug_row'].replace(to_replace=drug_idx_dict)
    summary_df.loc[:, 'did_col'] = summary_df.loc[:, 'drug_col'].replace(to_replace=drug_idx_dict)
    summary_df.loc[:, 'cid'] = summary_df.loc[:, 'cell_line_name'].replace(to_replace=cell_idx_dict)

    cell_df = cell_df[cell_df['name'].isin(cell_list)]
    drug_df = drug_df[drug_df['dname'].isin(drug_list)]
    cell_df.loc[:, 'cid'] = cell_df.loc[:, 'name'].replace(to_replace=cell_idx_dict)
    drug_df.loc[:, 'did'] = drug_df.loc[:, 'dname'].replace(to_replace=drug_idx_dict)
    
    # get stitich, ttd, drugbank target data
    print(f'process drugbank, ttd, stitch target data')
    if pathlib.Path(data_dict['stitch_target']).is_file():
        stitch_df = pd.read_pickle(f'{args.root}/stitch_target.pkl')
    else:
        stitch = srcSTITCHParser.ParseSTITCH(root=args.root)
        stitch_df = stitch.get_processed_data() # cid, pid, symbol

    drugbank = srcDrugBankParser.ParseDrugBank(root=args.root)
    drugbank_df = drugbank.get_processed_data() # drugbank_id, gene
    if pathlib.Path(data_dict['ttd_target']).is_file():
        ttd_df = pd.read_pickle(f'{args.root}/ttd_target.pkl')
    else:
        ttd = srcTTDParser.ParseTTD(root=args.root)
        ttd_df = ttd.get_processed_data()

    # add did
    cidm_df = pd.merge(stitch_df, drug_df[['cid_m', 'did']],
                       left_on='cid', right_on='cid_m', how='inner')
    cids_df = pd.merge(stitch_df, drug_df[['cid_s', 'did']],
                       left_on='cid', right_on='cid_s', how='inner')
    stitch_target_df = pd.concat([cidm_df, cids_df], axis=0)
    stitch_target_df = stitch_target_df.drop_duplicates(subset=['cid', 'pid'])[['did', 'symbol']]

    drugbank_id_list = sorted(list(set(drugbank_df['drugbank_id'])&set(drug_df['drugbank_id'])))
    drugbank_df = drugbank_df[drugbank_df['drugbank_id'].isin(drugbank_id_list)]
    drugbank_target_df = pd.merge(drugbank_df, drug_df[['drugbank_id', 'did']],
                                  left_on='drugbank_id', right_on='drugbank_id', how='inner')
    drugbank_target_df = drugbank_target_df[['did', 'Gene']]
    drugbank_target_df.columns = ['did', 'symbol']

    drug_df = drug_df.dropna(subset=['cid'], axis=0)
    drug_df['cid'] = drug_df['cid'].astype(int).astype(str)
    cid_list = sorted(list(set(ttd_df['cid'].values.tolist())&set(drug_df['cid'].values.tolist())))
    ttd_df = ttd_df[ttd_df['cid'].isin(cid_list)]
    ttd_target_df = pd.merge(ttd_df, drug_df[['cid', 'did']],
                             left_on='cid', right_on='cid', how='inner')
    ttd_target_df = ttd_target_df[['did', 'symbol']]

    # combine all recources
    target_df = pd.concat([stitch_target_df, drugbank_target_df, ttd_target_df])
    target_df = target_df.drop_duplicates(keep='first')
    target_df = long2wide(target_df, debug=False) # gene by did
    stat_df = target_df.sum(axis=0).to_frame(name='target size')
    print(f'distribution of drug target size')
    print(stat_df.describe())

    # find overlapping did
    did = [set(target_df.columns.tolist()), set(summary_df['did_row']), set(summary_df['did_col']), set(drug_df['did'])]
    did_list = set.intersection(*did)

    # subsetting to include intersection dids
    drug_df = drug_df[drug_df['did'].isin(did_list)]
    target_df = target_df[sorted(did_list)]
    rowfilter = (summary_df.did_col.isin(did_list)) & (summary_df.did_row.isin(did_list))
    summary_df = summary_df[rowfilter]
    

    ########################################################################################
    # Save to files
    # drugcomb_processed_summary.pkl
    # drugcomb_processed_drug.pkl
    # drugcomb_processed_cell.pkl
    # drugcomb_processed_target.pkl
    ########################################################################################
    # create triplet and index combo columns
    summary_df['triplet'] = summary_df['drug_row']+'_'+summary_df['drug_col']+'_'+summary_df['cell_line_name']
    summary_df['icombo'] = summary_df['did_row']+'='+summary_df['did_col']+'='+summary_df['cid']+'=r'+summary_df['conc_r'].astype(str)+'=c'+summary_df['conc_c'].astype(str)
    summary_df['ipair'] = summary_df['did_row']+'-'+summary_df['did_col']

    # add dosage region
    rule_r = [
                (summary_df['conc_r'] >= summary_df['ic50_row']),
                (summary_df['conc_r'] < summary_df['ic50_row']),
             ]

    rule_c = [
                (summary_df['conc_c'] >= summary_df['ic50_col']),
                (summary_df['conc_c'] < summary_df['ic50_col'])
            ]
    # define results
    results = ['high', 'low']
    summary_df['cate_r'] = np.select(rule_r, results)
    summary_df['cate_c'] = np.select(rule_c, results)
    def add_region(row):
        if row['cate_r'] == 'high' and row['cate_c'] == 'high':
            return 'high'
        elif row['cate_r'] == 'high' and row['cate_c'] == 'low':
            return 'high_low'
        elif row['cate_r'] == 'low'  and row['cate_c'] == 'high':
            return 'low_high'
        return 'low'
    summary_df['dose_region'] = summary_df.apply(lambda row: add_region(row), axis=1)
    print(summary_df[['icombo', 'synergy_loewe', 'dose_region', 'ic50_row', 'ic50_col', 'drug_row_clinical_phase', 'drug_col_clinical_phase']])
    # save files to folder
    fout = args.fout_path
    pathlib.Path(fout).mkdir(parents=True, exist_ok=True)
    summary_df.to_pickle(f'{fout}/drugcomb_processed_summary.pkl')
    drug_df.to_pickle(f'{fout}/drugcomb_processed_drug.pkl')
    cell_df.to_pickle(f'{fout}/drugcomb_processed_cell.pkl')
    target_df.to_pickle(f'{fout}/drugcomb_processed_target.pkl')

    print(f'Data Processing Completed. Find Processed Data at: {fout}')
    print(f'saving files')
    print(f'{fout}/drugcomb_processed_summary.pkl')
    print(f'{fout}/drugcomb_processed_drug.pkl')
    print(f'{fout}/drugcomb_processed_cell.pkl')
    print(f'{fout}/drugcomb_processed_target.pkl')
