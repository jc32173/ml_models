
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

#import warnings
#warnings.filterwarnings('ignore')

import multiprocessing as mp
#from multiprocessing import Process
from rdkit import Chem
import pickle as pk
import pandas as pd
import numpy as np
import math
import sys, os
from datetime import datetime
import json
import logging


# Global logger:
logging.basicConfig(level=logging.WARNING)

# Set up logger for module:
logger = logging.getLogger(__name__)
# Set logging levels, especially when debugging:
logging.getLogger().setLevel(logging.WARNING)


# Set random seeds to make results fully reproducible:
if len(sys.argv) >= 3:
    rand_seed = int(sys.argv[2])
    np.random.seed(rand_seed+123)
else:
    rand_seed = None

# Update this when code is moved to programs folder:
sys.path.insert(0, '/users/xpb20111/programs/deepchem_dev_nested_CV')

# Import code:
from training_utils.record_results import setup_results_series
from training_utils.splitting import nested_CV_splits, train_test_split, check_dataset_split
from sklearn_models.build_models.training import score_model
from sklearn_models.build_models.metrics import all_metrics
from cheminfo_utils.calc_desc import calc_desc

#import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn_models.models.models import *
from sklearn_models.build_models.preprocess import GetDataset

#all_metrics ....

# Hardcode standard filenames:
train_val_test_split_filename = 'train_val_test_split.csv'
#all_model_results_filename = 'GCNN_info_all_models.csv'


# Read input details:
json_infile = sys.argv[1]
print('Reading input parameters from .json file: {}'.format(json_infile))
run_input = json.load(open(json_infile, 'r'))

# Import model:
#importlib.import_module(run_input['training']['model_fn_str'])

# Choose metrics for classification or regression:
all_metrics = all_metrics[run_input['dataset']['mode']]

# Generate empty Series object to store results:
run_results = setup_results_series(run_input, all_metrics)


# ==============================================
# Read any results from previous incomplete runs
# ==============================================

## File to save model details for each set of hyperparameters:
#if not os.path.isfile(all_model_results_filename):
#    df_prev_runs = None
#    info_out = open(all_model_results_filename, 'w')
#    info_out.write(';'.join(run_results.index.get_level_values(0))+'\n')
#    info_out.write(';'.join(run_results.index.get_level_values(1))+'\n')
#    # Need to flush here to ensure header is only written once and removed 
#    # from buffer, otherwise header is written before the results from each model:
#    info_out.flush()
#    #mod_i = 0
#
## Read data from any previous runs:
#else:
#    df_prev = pd.read_csv(all_model_results_filename, sep=';', header=[0, 1])
#    df_prev_runs = df_prev[[('model_info', 'resample_number'), 
#                            ('model_info', 'cv_fold')] + \
#                           [('hyperparams', hp_name) 
#                            for hp_name in df_prev['hyperparams'].columns]]
#    if len(set(df_prev.columns) & set(run_results.index)) != len(df_prev.columns):
#        raise ValueError('')
#    #mod_i = df_prev[('model_info', 'model_number')].max() + 1
#    info_out = open(all_model_results_filename, 'a')
#    # Check order of output matches existing column order:
#    run_results = run_results[df_prev.columns]

# Keep an empty copy to revert to after each run:
run_results_empty = run_results.copy()


# ===============
# Load dataset(s)
# ===============

print('Loading dataset: {}'.format(run_input['dataset']['dataset_file']))

# Load dataset:
load_data = GetDataset(**run_input['dataset'], 
                       **run_input['preprocessing'])
full_dataset, additional_params = load_data()

# Load external test sets if given:
ext_test_set = {}
if ('ext_datasets' in run_input) and len(run_input['ext_datasets']) > 0:
    for set_name in run_input['ext_datasets']['_order']:
        set_info = run_input['ext_datasets'][set_name]
        print('Loading external dataset: {} ({})'.format(set_info['dataset_file'], set_name))
        ext_test_set[set_name], _ = load_data(**set_info)

print('Dataset shape: X: {}, y: {}'.format(
    full_dataset.X.shape, full_dataset.y.shape))


# =============
# Split dataset
# =============

df_split_ids = nested_CV_splits(train_val_test_split_filename,
                                full_dataset.ids,
                                run_input,
                                rand_seed=rand_seed)

# =======================
# Train individual models
# =======================

n_cpus = run_input['training'].get('n_cpus')
#pool = mp.Pool(n_cpus)
#print('Training separate models on {} CPUs'.format(pool._processes))

# Add index number for selecting datasets:
#df_split_ids.loc[x.index, 'idx'] = range(len(full_dataset.ids))
#df_split_ids['idx'] = df_split_ids['idx'].astype(int)

df_final_results = pd.DataFrame()
for resample_n in range(run_input['train_test_split']['n_splits']):
    run_results = run_results.copy()
    run_results[('model_info', 'resample_number')] = resample_n

    train_val_ids = df_split_ids.loc[np.any(df_split_ids[str(resample_n)] != 'test', axis=1)].index
    train_val_set = full_dataset.loc[train_val_ids]
    test_set = full_dataset.loc[~df_split_ids.index.isin(train_val_ids)]

    # Get train/val splits for hyperparameter tuning:
    hyper_cv_splits = df_split_ids.loc[train_val_ids, str(resample_n)]
    hyper_cv_splits.loc[train_val_ids, 'idx'] = range(len(train_val_set))
    hyper_cv_splits['idx'] = hyper_cv_splits['idx'].astype(int)
    hyper_cv_splits = \
    [[hyper_cv_splits.set_index(str(cv_n)).loc['train', 'idx'].to_list(),
      hyper_cv_splits.set_index(str(cv_n)).loc['val', 'idx'].to_list()]
     for cv_n in range(run_input['train_val_split']['n_splits'])]

    model = eval(run_input['training']['model_fn_str'])


    # =====================
    # Hyperparameter tuning
    # =====================

    hyperparams = {hp: run_input['hyperparams'][hp] for hp in run_input['hyperparams']['_order']}
    
    # Exhaustive grid search:
    if 'hyperparam_search' not in run_input['training'] or \
       run_input['training'].get('hyperparam_search') == 'grid':
        model = GridSearchCV(estimator=model,
                             param_grid=hyperparams, 
                             n_iter=run_input['training']['n_iter'],
                             cv=hyper_cv_splits,
                             refit=True,
                             verbose=1,
                             n_jobs=n_cpus)

    # Random search:
    elif run_input['training']['hyperparam_search'] in ['rand', 'random']:
        model = RandomizedSearchCV(estimator=model,
                                   param_distributions=hyperparams, 
                                   n_iter=run_input['training']['n_iter'],
                                   random_state=rand_seed+resample_n,
                                   cv=hyper_cv_splits,
                                   refit=True,
                                   verbose=1,
                                   n_jobs=n_cpus)

    model.fit(train_val_set.X,
              train_val_set.y.squeeze())

    # Score model:
    score_model(model,
                train_set=train_val_set,
                test_set=test_set,
                ext_test_set={},
                all_metrics=all_metrics,
                mode=run_input['dataset']['mode'],
                run_results=run_results)

    df_final_results = df_final_results.append(pd.DataFrame(run_results).T)

    print('Finished training resample: {}'.format(resample_n))


# =======================
# Get average performance
# =======================

# Get average, standard deviation and standard error:
df_av_results = pd.DataFrame()
for dataset in ['train', 'test']:
    df_agg = df_final_results[dataset].agg(['mean', 'std', 'sem']).T
    df_agg.index.rename('metric', inplace=True)

    # Calculate confidence intervals:
    df_agg.loc['rmsd', 'CI95_lower'] = df_agg.loc['rmsd', 'mean']-1.69*df_agg.loc['rmsd', 'sem']
    df_agg.loc['rmsd', 'CI95_upper'] = df_agg.loc['rmsd', 'mean']+1.69*df_agg.loc['rmsd', 'sem']

    df_agg['dataset'] = dataset
    df_agg = df_agg.set_index('dataset', append=True)\
                   .reorder_levels(['dataset', 'metric'])

    df_av_results = df_av_results.append(df_agg)

df_av_results.to_csv(run_input['training']['model_fn_str']+'_info_av_performance.csv')

print('Average performance on test set:')
print(df_av_results.loc['test'].to_string(index=True))

