"""
Parse TTD
"""

import pathlib
import pandas as pd

class ParseTTD:
    """
    """
    def __init__(self, root=None , debug=False):
        self.debug = debug

        f_dict = {'target': 'P1-01-TTD_target_download.txt',
                  'drug': 'P1-03-TTD_crossmatching.txt',
                  'mapping': 'P1-07-Drug-TargetMapping.xlsx'}

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

    def clean_map(self):
        df = pd.read_excel(self.f_dict['mapping'])
         
        black_list = ['Terminated', 'Discontinued in Phase 1',
                      'Discontinued in Phase 1/2', 'Discontinued in Phase 2',
                      'Discontinued in Phase 3', 'Withdrawn from market',
                      'Discontinue in Phase 1 Trial', 'Discontinued in Phase 2a',
                      'Discontinued in Phase 2b', 'Discontinued in Phase 2/3',
                      'Discontinued in Preregistration', 'Discontinued in Phase 4']
        df = df[~df['Highest_status'].isin(black_list)]
        return df

    def clean_drug(self):
        df = pd.read_csv(self.f_dict['drug'], skiprows=39, header=None, sep="\t")[[0,1,2]]
        df.columns = ['drug_id', 'abbrev', 'name']
        df = df.dropna(how='all', axis=0) #drop empty rows


        # collect drug pubchemid
        result_list = []

        # loop through all drug id
        drug_list = list(df['drug_id'].unique())
        for drug in drug_list:
            s_df = df[df['drug_id']==drug]
            symbol_list = s_df[s_df['abbrev']=='PUBCHCID']['name'].values
            if len(symbol_list) > 0:
                if  "; " in symbol_list[0]:
                    symbol_list = symbol_list[0].split(';')
                    for symbol in symbol_list:
                        result_list.append( (drug, symbol) )
                elif "-" in symbol_list[0]:
                    symbol_list = symbol_list[0].split('-')
                    for symbol in symbol_list:
                        result_list.append( (drug, symbol) )
                else:
                    symbol = symbol_list[0]
                    result_list.append( (drug,symbol) )

        # list to df
        drug_df = pd.DataFrame.from_records(result_list, columns=['ttd_id', 'cid'])
        drug_df = drug_df.drop_duplicates(keep='first')
        return drug_df

    def clean_target(self):
        df = pd.read_csv(self.f_dict['target'], skiprows=39, header=None, sep="\t")[[0,1,2]]
        df.columns = ['target_id', 'abbrev', 'name']
        df = df.dropna(how='all', axis=0) #drop empty rows

        # collect target gene name
        result_list = []

        # loop through all target id to get gene name
        target_list = list(df['target_id'].unique())
        for target in target_list:
            s_df = df[df['target_id']==target]
            symbol_list = s_df[s_df['abbrev']=='GENENAME']['name'].values
            if len(symbol_list) > 0:
                if  "; " in symbol_list[0]:
                    symbol_list = symbol_list[0].split(';')
                    for symbol in symbol_list:
                        result_list.append( (target, symbol) )
                elif "-" in symbol_list[0]:
                    symbol_list = symbol_list[0].split('-')
                    for symbol in symbol_list:
                        result_list.append( (target, symbol) )
                else:
                    symbol = symbol_list[0]
                    result_list.append( (target,symbol) )

        # list to df
        target_df = pd.DataFrame.from_records(result_list, columns=['target', 'symbol'])
        target_df = target_df.drop_duplicates(keep='first')
    
        return target_df

    def get_processed_data(self):
        map_df = self.clean_map() # TargetID, DrugID, Highest_status, MOA
        drug_df = self.clean_drug() # ttd_id, cid
        target_df = self.clean_target() # target, symbol
        
        rowfilter = (map_df['TargetID'].isin( target_df['target'].values.tolist() )) & \
                    (map_df['DrugID'].isin( drug_df['ttd_id'].values.tolist()  ))
        map_df = map_df[rowfilter]

        # add cid
        d_df = pd.merge(map_df, drug_df, 
                        left_on='DrugID', right_on='ttd_id', how='inner')
        d_df = d_df.drop(columns=['DrugID', 'Highest_status', 'MOA'])
        # add symbol
        t_df = pd.merge(d_df, target_df,
                        left_on='TargetID', right_on='target', how='inner')
        t_df = t_df.drop(columns=['TargetID'])
        return t_df #[ttd_id cid, target, symbol] 
