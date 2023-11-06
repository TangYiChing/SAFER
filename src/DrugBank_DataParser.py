"""
Parse DrugBank
"""
import pathlib
import pandas as pd

class ParseDrugBank:
    """
    """
    def __init__(self, root=None , debug=False):
        self.debug = debug

        f_dict = {'db': 'drug_DrugBank_target.csv'}

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


    def get_processed_data(self, debug=False):
        """return drug target data"""
        # retrieve data
        df = pd.read_csv(self.f_dict['db'], header=0)
        df = df[['drugbank_id', 'Gene']]
        return df
