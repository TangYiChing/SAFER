# SAFER: sub-hypergraph attention-based neural network for predicting effective responses to dose combinations

# Setup:
```bash
$pip install -r requirements.txt
```
# Download processed datasets from Zenodo
```bash
$pip install zenodo-get
$zenodo_get 10.5281/zenodo.10076432
```

# Cross-validation of SAFER-C2 model on the O'Neil dataset:
```bash
$python ./src/trainCV_model.py -processed ./data/input_data/c2/ -train ./data/cv_data/ONEIL/train/ -valid ./data/cv_data/ONEIL/valid/ -test ./data/cv_data/ONEIL/test/ -onehot -t classification -param ./data/param_data/cv10.best_param.pkl -k 10 -g 6 -p safer-c2 -fout ./
```
# Cross-validation of SAFER-C3 model on the O'Neil dataset:
```bash
$python ./src/trainCV_model.py -processed ./data/input_data/c3/ -train ./data/cv_data/ONEIL/train/ -valid ./data/cv_data/ONEIL/valid/ -test ./data/cv_data/ONEIL/test/ -onehot -t classification -param ./data/param_data/cv10.best_param.pkl -k 10 -g 6 -p safer-c3 -fout ./
```
# External validation on the non-overlapping Almanac dataset:
```bash
$python ./src/external_validate.py -processed ./data/input_data/c2/ -train ./data/cv_data/ONEIL/train/ -valid ./data/cv_data/ONEIL/valid/ -test ./data/cv_data/ONEIL/test/ -external ./data/external_data/ALMANAC.non_overlapping_triplets.pkl -onehot -t classification -mdl ./model/SAFER-C2.best_model.pt -k 10 -g 6 -p safer-c2 -fout ./
```
```bash
$python ./src/external_validate.py -processed ./data/input_data/c3/ -train ./data/cv_data/ONEIL/train/ -valid ./data/cv_data/ONEIL/valid/ -test ./data/cv_data/ONEIL/test/ -external ./data/external_data/ALMANAC.non_overlapping_triplets.pkl -onehot -t classification -mdl ./model/SAFER-C3.best_model.pt -k 10 -g 6 -p safer-c3 -fout ./
```
# Data pre-processing 
### NOTE: Skip this step if you already download the processed data available in the Zenodo repository: 10.5281/zenodo.10076432
```bash
$python ./src/process_raw_data.py -r ./database/ -fout ./data/processed_data/

# Prepare input data for SAFER-C2
$python ./src/generate_hypergraph_subgraph.py -r ./database/ -processed ./data/processed_data/ -s smile -k 9 -sg -m c2 -c exp -dose -clinical -fout ./data/input_data/c2/

# Prepare input data for SAFER-C3
python ./src/generate_hypergraph_subgraph.py -r ./database/ -processed ./data/processed_data/ -s smile -k 9 -sg -m c3 -c exp -dose -clinical -fout ./data/input_data/c3/
```
