
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


# Set up logger for module:
logger = logging.getLogger(__name__)
# Set logging levels, especially when debugging:
logging.getLogger().setLevel(logging.ERROR)
#logging.getLogger().setLevel(logging.DEBUG)


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
from training_utils.splitting import train_test_split, check_dataset_split

all_metrics ....

# Hardcode standard filenames:
train_val_test_split_filename = 'train_val_test_split.csv'
#all_model_results_filename = 'GCNN_info_all_models.csv'


# Read input details:
json_infile = sys.argv[1]
print('Reading input parameters from .json file: {}'.format(json_infile))
run_input = json.load(open(json_infile, 'r'))

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


# =============
# Split dataset
# =============

df_split_ids = split_dataset(train_val_test_split_filename, 
                             run_input)


# =====================
# Hyperparameter tuning
# =====================

## Exhaustive grid search:
#if 'hyperparam_search' not in run_input['training'] or \
#   run_input['training'].get('hyperparam_search') == 'grid':
#    get_hyperparam_iter, n_hyperparams = get_hyperparams_grid(run_input['hyperparams'])
#
## Random search:
#elif run_input['training']['hyperparam_search'] in ['rand', 'random']:
#    get_hyperparam_iter, n_hyperparams = \
#        get_hyperparams_rand(hyperparams=run_input['hyperparams'],
#                             n_iter=run_input['training'].get('hyperparam_search_iterations'))
#
## GP:
#elif run_input['training']['hyperparam_search'] == 'gp':
#    raise NotImplementedError('GP not yet implemented')


# =======================
# Train individual models
# =======================

n_cpus = run_input['training'].get('n_cpus')
pool = mp.Pool(n_cpus)
print('Training separate models on {} CPUs'.format(pool._processes))

model_results = []

# Add index number for selecting datasets:
df_split_ids.loc[full_dataset.ids, 'idx'] = range(len(full_dataset.ids))
df_split_ids['idx'] = df_split_ids['idx'].astype(int)

for resample_n in df_split_ids.drop(columns=['idx']).columns:
#    if cv_n == 'refit':
#        continue


    mod = GridSearchCV





    split_idxs = df_split_ids.set_index((resample_n, cv_n))['idx']
    hyperparam_iter = get_hyperparam_iter()
    for mod_i, hp_dict in enumerate(hyperparam_iter):

        # Check whether set of hyperparameters have already been run:
        if df_prev_runs and \
           np.all(np.array([(df_prev_runs[('model_info', 'resample_number')] == resample_n) & \
                            (df_prev_runs[('model_info', 'cv_fold')] == cv_n)] + \
                           [df_prev_runs[('hyperparams', hp_name)] == hp_val 
                            for hp_name, hp_val in hp_dict.items()]),
                  axis=1) > 0:
            continue

        print('Submitting resample: {}, cv fold: {}, hyperparameter combination: {}'.format(resample_n, cv_n, mod_i))

        setup_training(resample_n, cv_n, mod_i,
                       full_dataset, split_idxs,
                       hp_dict, run_input,
                       run_results, model_results)


# ==================
# Refit final models
# ==================

# Fit best model for each resample:
df_cv_results = pd.DataFrame(columns=run_results.index)
refit_models = []
for model_result in model_results:
    df_model_result = model_result.get()
    resample_n, cv_n, mod_i = df_model_result['model_info'][['resample_number', 'cv_fold', 'model_number']]
    print('Finished training resample: {}, cv fold: {}, hyperparameter combination: {}'.format(resample_n, cv_n, mod_i))
    df_cv_results = df_cv_results.append(df_model_result, ignore_index=True)

    if df_cv_results.loc[df_cv_results[('model_info', 'resample_number')] == resample_n, 
                         [('model_info', 'cv_fold')]]\
                    .nunique()\
                    .squeeze() == run_input['splitting']['inner_split']['n_splits'] \
       and \
       np.all(df_cv_results.loc[df_cv_results[('model_info', 'resample_number')] == resample_n]\
                           .groupby(('model_info', 'cv_fold'))\
                           .count()[('model_info', 'model_number')] == n_hyperparams):
        hp_cols = [('hyperparams', hp_name) for hp_name in df_cv_results['hyperparams'].columns]
        best_hp_idx = df_cv_results.loc[df_cv_results[('model_info', 'resample_number')] == resample_n]\
                                   .astype({hp_col : str for hp_col in hp_cols})\
                                   .groupby([('hyperparams', hp_name) for hp_name in df_cv_results['hyperparams'].columns])\
                                   .mean()\
                                   .sort_values(('val', 'loss'))\
                                   .iloc[[0]]\
                                   .index

        best_hp_dict = pd.Series(data=best_hp_idx.values.squeeze(),
                                 index=pd.MultiIndex.from_tuples(best_hp_idx.names))\
                         ['hyperparams']\
                         .apply(eval)\
                         .to_dict()

        cv_n = 'refit'

        split_idxs = df_split_ids.set_index((resample_n, cv_n))['idx']
        print('Submitting resample: {}, cv fold: {}, hyperparameter combination: {}'.format(resample_n, cv_n, mod_i))
        
        setup_training(resample_n, cv_n, mod_i,
                       full_dataset, split_idxs,
                       best_hp_dict, run_input,
                       run_results, refit_models)

df_final_results = pd.DataFrame(columns=run_results.index)
for model_result in refit_models:
    df_model_result = model_result.get()
    resample_n, cv_n, mod_i = df_model_result['model_info'][['resample_number', 'cv_fold', 'model_number']]
    print('Finished training resample: {}, cv fold: {}, hyperparameter combination: {}'.format(resample_n, cv_n, mod_i))
    df_final_results = df_final_results.append(df_model_result, ignore_index=True)
pool.close()

df_final_results.to_csv('GCNN_info_refit_models.csv', index=False)

## Get average, standard deviation and standard error
#df_av_results = \
#df_final_results.drop(columns=['hyperparams'])\
#                .agg()
#                .to_csv('GCNN_average_results.csv')
#
#print('Average performance on test set:')
#print(df_final_results['test'].mean().to_string(index=True))
