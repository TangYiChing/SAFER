"""
Fit model with the best parameter
"""

import argparse
import time
import random
import pickle
import pathlib
import numpy as np
import pandas as pd
import sklearn.feature_extraction as skfet
import sklearn.preprocessing as skpre

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import MSELoss, L1Loss, CrossEntropyLoss, BCEWithLogitsLoss, BCELoss, HuberLoss
from torch.optim import Adam, Adagrad

import optuna
from optuna.trial import TrialState

import utility as srcUtil
import SynergyDataset as srcData
import SynergyNet as srcNet
import FitSynergyNet as srcFit
import pytorchtools as srcTCH

def parse_parameter():
    parser = argparse.ArgumentParser(description="Validate multi-modal multi-task model")
    parser.add_argument("-processed","--processed_path",
                        required = True,
                        help = "folder path to processed data files. e.g., ./data/processed_data/")
    parser.add_argument("-train", "--train_data",
                        required = True,
                        help = "file name of train data.")
    parser.add_argument("-valid", "--valid_data",
                        required = False,
                        help = "file name of valid data.")
    parser.add_argument("-test", "--test_data",
                        required = True,
                        help = "file name of test data.")
    parser.add_argument("-i", "--inductive_int",
                        required = False,
                        type = int,
                        default = 0,
                        help = "use p of training samples' subject edges as inductive network. default: 100 percent. (set 0 to turn off inductive mode)")
    parser.add_argument("-onehot", "--onehot_data",
                        action = "store_true",
                        help = "add dosing onehot data if enabled")
    parser.add_argument("-t", "--task_str",
                        default = 'classification',
                        help = "classification task or regression task. default: classification")
    parser.add_argument("-param", "--param_path",
                        required = False,
                        help = "file name of  hyperparameter file, will traing model with the parameters if provided. (e.g., best_param.pkl)")
    parser.add_argument("-auxi", "--use_auxi",
                        action = "store_true",
                        help = "enable auxiliary task if True")
    parser.add_argument("-k", "--k_int",
                        required = False,
                        default = 10,
                        type = int,
                        help = "integer representing kfold. default: 10")
    parser.add_argument("-g", "--gpu_int",
                        required = False,
                        default = 0,
                        type = int,
                        help = "integer representing gpu. default: 0")
    parser.add_argument("-s", "--seed",
                        required = False,
                        default = 42,
                        type = int,
                        help = "seed number for reproducing results")
    parser.add_argument("-p", "--prefix_str",
                        required = True,
                        help = "prefix of output files")
    parser.add_argument("-fout", "--fout_path",
                        required = True,
                        help = "folder path to store result")
    return parser.parse_args()

def fit_predict(ARGS, data_dict, hypergraph_dict, subgraph_dict):
    # set arguments & seed
    args = ARGS
    srcUtil.set_seed(args.seed)

    # set hyperparameters
    if args.param_path is not None:
        # load from file
        print(f'using parameters from file:')
        with open(args.param_path, 'rb') as f:
            param_dict = pickle.load(f)

        PARAM_DICT = {'n_hid': param_dict['n_hid'],
                      'dropout': param_dict['dropout'],
                      'fc_dropout': param_dict['fc_dropout']}
                      #'lr': param_dict['lr'],
                      #'lr_decay': param_dict['lr_decay'],
                      #'weight_decay': param_dict['weight_decay']}
    else:
        print(f'using default parameters:')
        PARAM_DICT = {'n_hid': 200,
                      'dropout': 0.2,
                      'fc_dropout': 0.4}
                      #'lr': 0.001,
                      #'lr_decay': 0.001,
                      #'weight_decay': 0.001}
    for k, v in PARAM_DICT.items():
        print(f'{k}={v}')

    # assign class weight
    DEVICE = torch.device("cuda:"+str(args.gpu_int) if torch.cuda.is_available() else "cpu")
    region_count = []
    pos_weight = 0.0
    n_pos = 0
    n_neg = 0
    for region in [0,1,2,3]:
        sample = data_dict['train'][data_dict['train']['region_label']==region]
        region_count.append(len(sample))

        if args.task_str == 'classification':
            #pos_weight += (sample['synergy_label']==0).sum() / (sample['synergy_label']==1).sum() # pos_weight > 1, improve recall
            n_pos += (sample['synergy_label']==1).sum()
            n_neg += (sample['synergy_label']==0).sum()

    if args.task_str == 'classification':
        pos_weight = n_neg / n_pos # pos_weight < 1, improve precision
        synergy_weight = torch.tensor([pos_weight], dtype=torch.float)
        loss_fn = BCEWithLogitsLoss(pos_weight=synergy_weight.to(DEVICE)) # for synergy prediction
    else:
        loss_fn = HuberLoss() #MSELoss() # for synergy prediction
     
    region_weight = 1.0 / torch.tensor(region_count, dtype=torch.float)
    region_weight = region_weight / region_weight.sum() 
    loss_fn2 = CrossEntropyLoss(weight=region_weight.to(DEVICE)) # for region prediction   
    print(f'weighted loss')
    print(f'    pos_weight={pos_weight} | region_weight={region_weight}')



    # set model and training parameters
    N_EPOCHS = 800
    BATCH_SIZE = 128
    N_PATIENCES = 10
    out_syn_n = 1
    out_region_n = 4
    in_dose_n = hypergraph_dict['dose'][0].shape[0] 
    in_chemical_n = hypergraph_dict['chemical'][0].shape[0]
    if args.onehot_data == True:
        in_onehot_n = subgraph_dict['onehot'].shape[1]
    else:
        in_onehot_n = None

    model = srcNet.DrugCombHypergraph(in_chemical_n=in_chemical_n,
                                      in_dose_n=in_dose_n,
                                      in_onehot_n=in_onehot_n,
                                      out_syn_n=out_syn_n,
                                      out_region_n=out_region_n,
                                      n_hid=PARAM_DICT['n_hid'],
                                      dropout=PARAM_DICT['dropout'],
                                      fc_dropout=PARAM_DICT['fc_dropout'],
                                      join=False,
                                      device=DEVICE,
                                      use_auxi=args.use_auxi)
    model = model.cuda(torch.cuda.set_device(DEVICE))
    
 
    lr = 0.0001
    lr_decay = 0.00001
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=lr_decay,
                                 betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min = 1e-7)



    # 2. set fitting parameters
    dose_H = hypergraph_dict['dose'][0]
    dose_G = hypergraph_dict['dose'][1]
    chemical_H=hypergraph_dict['chemical'][0]
    chemical_G=hypergraph_dict['chemical'][1]
    fit_data = srcFit.DrugCombHypergraph(chemical_H=chemical_H,
                                         chemical_G=chemical_G,
                                         dose_H=dose_H,
                                         dose_G=dose_G,
                                         batch_size=BATCH_SIZE,
                                         optimizer=optimizer,
                                         scheduler=scheduler,
                                         loss_fn=loss_fn,
                                         loss_fn2=loss_fn2,
                                         device=DEVICE,
                                         task=args.task_str,
                                         nratio=1,
                                         use_auxi=args.use_auxi,
                                         use_onehot=args.onehot_data) # True/False

    # 1. transform data
    print(f'normalize data split')
    subgraph_split_dict = transform_data(subgraph_dict, data_dict)

    # 2. create data batches
    train_data = srcData.DrugCombDataset(summary_data=data_dict['train'],
                                         chemical_subgraph=subgraph_split_dict['chemical']['train'],
                                         dose_subgraph=subgraph_split_dict['dose']['train'],
                                         onehot_subgraph=subgraph_split_dict['onehot']['train'],
                                         shuffle=True)
    test_data = srcData.DrugCombDataset(summary_data=data_dict['test'],
                                         chemical_subgraph=subgraph_split_dict['chemical']['test'],
                                         dose_subgraph=subgraph_split_dict['dose']['test'],
                                         onehot_subgraph=subgraph_split_dict['onehot']['test'],
                                         shuffle=False)

    # fit
    early_stopping = srcTCH.EarlyStopping(patience=N_PATIENCES, path=args.fout_path+'/'+args.prefix_str+'.early_stop.checkpoint.pt', verbose=True)
    for epoch in range(0, N_EPOCHS):
        model, train_loss, train_score = fit_data.fit(model, train_data)
        early_stopping(train_loss, model)
        if early_stopping.early_stop:
            print(f'Early Stop')
            break
    # predict
    y_dict, emb_dict = fit_data.predict(model, test_data)
    return y_dict, emb_dict, model



def subsampling_train(df, task_str):
    """
    :param df: dataframe
    :param task_str: string representing task type [classification, regression]
    :return df: subset of the original dataframe 
    """
    df_list = [] # sampling from each dose region
    if task_str == 'classification':
        # sampling from each dose region
        for region in [0,1,2,3]:
            sample = df[df['region_label']==region]
            imratio = (sample['synergy_label']==1).sum() / len(sample)
            sample = srcUtil.sampling_pos(sample, 'synergy_label', r=imratio) 
            df_list.append(sample)
        # merge all
        df = pd.concat(df_list, axis=0)

        # resampling from all samples
        #df = srcUtil.sampling(df, 'synergy_label', r=2)
    
    elif task_str == 'regression':
        for region in [0,1,2,3]:
            sample = df[df['region_label']==region]
            sample = sample.sample(frac=0.7).reset_index(drop=True)
            df_list.append(sample)
        # merge all
        df = pd.concat(df_list, axis=0)
    else:
        raise ValueError(f'Error, task_str is either classification or regression, but got {task_str}!!!')
    return df

def use_inductive(hypergraph_dict, subgraph_dict, train_df, inductive_int):
    """
    :param hypergraph_dict: dictionary of hypergraph data, keys=[dose, chemical]
    :param subgraph_dict: dictionary of subgraph data, keys=[dose, chemical]
    :param data_dict: dictionary of train data to be used for inductive learning
    :return hypergraph_dict:
    """
    if inductive_int > 0: # append subgraph to hypergraph
        print(f'inductive mode: {inductive_int}% of training subjects')
        #icombo_list = sorted( train_df.sample(frac=float(inductive_int)/100)['icombo'].unique() )
        cid_list = sorted( train_df.sample(frac=float(inductive_int)/100)['cid'].unique() )
        ipair_list = sorted( train_df.sample(frac=float(inductive_int)/100)['ipair'].unique() )
        for dt in ['dose', 'chemical']:
            if dt == 'dose':
                icombo_list = sorted(train_df[train_df['cid'].isin(cid_list)]['icombo'].unique())
            elif dt == 'chemical':
                icombo_list = sorted(train_df[train_df['ipair'].isin(ipair_list)]['icombo'].unique())
            else:
                icombo_list = sorted( train_df.sample(frac=float(inductive_int)/100)['icombo'].unique() )
            
            icombo_list = random.choices(icombo_list, k=int(inductive_int/100*len(icombo_list)))
            print(f'{dt}={dt}: inductive samples={len(icombo_list)}')
            H = hypergraph_dict[dt][0]
            subj_arr = subgraph_dict[dt].loc[:, icombo_list].values
            inductive_H = np.hstack([ H, subj_arr])
            inductive_H = srcUtil.transform_H(inductive_H)
            inductive_G = srcUtil.generate_G_from_H(inductive_H)
            hypergraph_dict[dt] = (inductive_H, inductive_G)
            # sanitycheck
            if np.isnan(inductive_G.sum().sum()):
                raise ValueError(f'Error, G contains nan!!!')
    elif inductive_int == 0: 
        print(f'transductive mode: transforming H, G')
        for dt in ['dose', 'chemical']:
            H, G = hypergraph_dict[dt]
            transformed_H = srcUtil.transform_H(H)
            transformed_G = srcUtil.generate_G_from_H(H)
            hypergraph_dict[dt] = (transformed_H, transformed_G)
    else:
        raise ValueError(f'Error, inductive_int should be an integer, but got {args.inductive_int}!!!')
    return hypergraph_dict

def transform_data(subgraph_dict, data_dict):
    """
    :param subgraph_dict: dictionary of subgraphs, keys=[dose, chemical]
    :param data_dict: dictionary of data, keys=[train, test]
    :return subgraph_dict: transformed subgraphs
    """
    subgraph_split_dict = {'dose':{'train':None, 'valid':None, 'test':None},
                           'chemical':{'train':None, 'valid':None, 'test':None},
                           'onehot':{'train':None, 'valid':None, 'test':None}}
    for dt in ['dose', 'chemical','onehot']:
        if dt in ['dose', 'chemical']:
            if subgraph_dict[dt] is not None:
                # select data split
                subg_df = subgraph_dict[dt]
                train_cols = sorted(set(subg_df.columns) & set(data_dict['train']['icombo']))
                test_cols = sorted(set(subg_df.columns) & set(data_dict['test']['icombo']))
                train_subg_df = subg_df[train_cols].T # subject by features
                test_subg_df = subg_df[test_cols].T # subject by features
                # transform data
                scaler = skpre.MinMaxScaler().set_output(transform="pandas").fit(train_subg_df)
                subgraph_split_dict[dt]['train'] = scaler.transform(train_subg_df).T # feature by subject
                subgraph_split_dict[dt]['test'] = scaler.transform(test_subg_df).T # feature by subject
        elif dt in ['onehot']:
            if subgraph_dict[dt] is not None:
                subg_df = subgraph_dict[dt]
                print(subg_df.head())
                train_idx = sorted(set(subg_df.index) & set(data_dict['train']['icombo']))
                test_idx = sorted(set(subg_df.index) & set(data_dict['test']['icombo']))
                train_subg_df = subg_df.loc[train_idx] # subject by features
                test_subg_df = subg_df.loc[test_idx] # subject by features
                # transform data
                scaler = skfet.text.TfidfTransformer().fit(train_subg_df)
                train_arr = scaler.transform(train_subg_df).toarray() # subject by feature
                test_arr = scaler.transform(test_subg_df).toarray() # subject by feature
                subgraph_split_dict[dt]['train'] = pd.DataFrame(train_arr, train_subg_df.index, train_subg_df.columns)
                subgraph_split_dict[dt]['test'] = pd.DataFrame(test_arr, test_subg_df.index, test_subg_df.columns)
        else:
            raise ValueError(f'data scaling: {dt} not supported!!!')
    return subgraph_split_dict



def train_cv(ARGS):
    args = ARGS
    """average over K folds"""
    eval_list = []
    best_score = 0

    # Step0: get data fold
    data_dict = {'train':None, 'test':None}
    for i in range(0,args.k_int):
        print(f'data fold={i}')
        train_fname = args.train_data+'/cv_'+str(i)+'.data.pkl'
        test_fname = args.test_data+'/cv_'+str(i)+'.data.pkl'
        data_dict['train'] = pd.read_pickle(train_fname)
        data_dict['test'] = pd.read_pickle(test_fname).sample(frac=1).reset_index(drop=True) 
        if data_dict['train'].equals(data_dict['test']):
            raise ValueError(f'Error, train and test data are the same!!!')
        if args.valid_data is not None:
            valid_fname = args.valid_data+'/cv_'+str(i)+'.data.pkl'
            valid_data = pd.read_pickle(valid_fname)
            data_dict['train'] = pd.concat([data_dict['train'], valid_data], axis=0)

        # transform y
        #if args.task_str == 'regression':
        #    reg_scaler = skpre.MinMaxScaler(feature_range=(0,1)).fit(data_dict['train']['synergy_label'].values.reshape(-1,1))
        #    data_dict['train']['synergy_label'] = reg_scaler.transform(data_dict['train']['synergy_label'].values.reshape(-1,1)) * 100
        #    data_dict['test']['synergy_label'] = reg_scaler.transform(data_dict['test']['synergy_label'].values.reshape(-1,1)) * 100
        #    print('train y={:}'.format(data_dict['train']['synergy_label'].describe()))
        #    print('test y={:}'.format(data_dict['test']['synergy_label'].describe()))

        # 0-1: subsampling train data
        data_dict['train'] = subsampling_train(data_dict['train'], args.task_str)

        # 0-2: load hypergraph and subgraph
        hypergraph_dict = {'dose': args.processed_path+'/dose_hypergraph.pkl',
                           'chemical': args.processed_path+'/chemical_hypergraph.pkl'}
        subgraph_dict = {'dose': args.processed_path+'/dose_subgraph.pkl',
                           'chemical': args.processed_path+'/chemical_subgraph.pkl'}
        for dt in ['dose', 'chemical']:
            # load hypergraph
            with open(hypergraph_dict[dt], 'rb') as fin:
                hypergraph_dict[dt] = pickle.load(fin)['hypergraph'] # a tuple of H,G
            # load subgraph
            subgraph_dict[dt] = pd.read_pickle(subgraph_dict[dt])

        # 0-3: choose inductive or transductive
        hypergraph_dict = use_inductive(hypergraph_dict, subgraph_dict, data_dict['train'], args.inductive_int)

        if args.onehot_data == True:
            print(f'use dosing onehot data')
            subgraph_dict['onehot'] = pd.read_pickle( args.processed_path+'/dosing_onehot.pkl' )
        else:
            subgraph_dict['onehot'] = None

        # 0-4: searching for optimal parameters over all folds
        y_dict, emb_dict, model = fit_predict(ARGS,data_dict, hypergraph_dict, subgraph_dict)

        # 0-5: evaluate model
        # inverse transform y
        #if args.task_str == 'regression':
        #    y_dict['y'] = reg_scaler.inverse_transform(y_dict['y'].reshape(-1,1)) / 100
        #    y_dict['y_pred'] = reg_scaler.inverse_transform(y_dict['y_pred'].reshape(-1,1)) / 100

        eval_df = srcUtil.eval_mts(y_dict['y'], y_dict['y_pred'], args.task_str)
        eval_df.columns = ['metrics', 'cv_'+str(i)]
        eval_df.set_index(['metrics'], inplace=True)
        eval_list.append(eval_df)
        eval_df.to_csv(args.fout_path+'/'+args.prefix_str+'.cv_'+ str(i) +'.evaluation_metrices.txt', header=True, index=True, sep="\t")
        
        # find best-performing model
        if args.task_str == 'classification':
            score = eval_df.loc['AUPRC', 'cv_'+str(i)] 
        else:
            score = eval_df.loc['Spearman r', 'cv_'+str(i)]
        if score >= best_score:
            best_score = score
            torch.save(model.state_dict(), args.fout_path+'/'+args.prefix_str+'best_model.pt')
            print(f'current best model=cv{i}: AUPRC={score}')

        # 0-6: save predictions
        pred_df = data_dict['test'][['icombo', 'synergy_loewe', 'synergy_label', 'region_label', 'dose_region']].copy()
        pred_df['fold'] = ['cv_'+str(i)] * len(pred_df)
        pred_df['predicted_synergy'] = y_dict['y_pred']
        #if args.task_str == 'regression':
            #pred = scaler.inverse_transform(pred.reshape(-1,1))
            #pred_df['synergy_label'] = pred_df['synergy_yj']
        if args.use_auxi:
            pred_df['predicted_region'] = y_dict['y2_pred']
            report = skmts.classification_report(pred_df['region_label'], pred_df['predicted_region'])
            print(f'performance of auxiliary task:\n{report}')
        pred_df.to_csv(args.fout_path+'/'+args.prefix_str+'.cv_'+ str(i) +'.model_prediction.txt', header=True, index=False, sep="\t")

        # 0-7: save subject embeddings
        #dose_cols = [ 'dose_'+str(i) for i in range(0, emb_dict['dose'].shape[1]) ]
        #dose_df = pd.DataFrame(emb_dict['dose'], index=pred_df['icombo'], columns=dose_cols)
        #chemical_cols = [ 'chem_'+str(i) for i in range(0, emb_dict['chemical'].shape[1]) ]
        #chemical_df = pd.DataFrame(emb_dict['chemical'], index=pred_df['icombo'], columns=chemical_cols)
        #emb_dict = {'dose':dose_df, 'chemical':chemical_df}
        #with open(args.fout_path+'/'+args.prefix_str+'.cv_'+ str(i) +'.subgraph_embeddings.pkl', 'wb') as f:
        #    pickle.dump(emb_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(args.fout_path+'/'+args.prefix_str+'.cv_'+ str(i) +'.subgraph_embeddings_attentionweights.pkl', 'wb') as f:
            pickle.dump(emb_dict, f, protocol=pickle.HIGHEST_PROTOCOL) # a tuple of (embeddings, attention_weights) for subjects

    

    # Step1: collect outputs
    eval_df = pd.concat(eval_list, axis=1)
    return eval_df
                
            
if __name__ == '__main__':
    # set parameters
    ARGS = parse_parameter()

    start = time.time()
    
    eval_df = train_cv(ARGS)
    print(eval_df) 

    end = time.time()
    spent = end-start
    print(f'Training completed in {spent//60:.0f}m {spent%60:.0f}s')
