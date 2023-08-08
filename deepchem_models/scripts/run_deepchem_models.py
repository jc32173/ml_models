
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf

import multiprocessing as mp
#from multiprocessing import Process
from rdkit import Chem
import deepchem as dc
import pickle as pk
import pandas as pd
import numpy as np
import math
import sys, os
from datetime import datetime
import json
import logging


# Set log level for tensorflow:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set up logger for module:
logger = logging.getLogger(__name__)
# Set logging levels, especially when debugging:
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
#logging.getLogger().setLevel(logging.DEBUG)
#logging.getLogger("tensorflow").setLevel(logging.DEBUG)


#loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
#for logger in loggers:
#    logger.setLevel(logging.DEBUG)
#logging.getLogger("tensorflow").setLevel(logging.ERROR)


# Set random seeds to make results fully reproducible:
if len(sys.argv) >= 3:
    rand_seed = int(sys.argv[2])
    np.random.seed(rand_seed+123)
    tf.random.set_seed(rand_seed+456)
else:
    rand_seed = None

# Update this when code is moved to programs folder:
sys.path.insert(0, '/users/xpb20111/programs/deepchem_dev_nested_CV')

# Import code:
from training_utils.record_results import setup_results_series
from training_utils.splitting import nested_CV_splits, train_test_split, check_dataset_split
from training_utils.model_scoring import get_average_model_performance
from deepchem_models.build_models.dc_metrics import all_metrics
from deepchem_models.build_models.dc_preprocess import GetDataset, apply_transforms, transform
from deepchem_models.build_models.dc_training import get_hyperparams_grid, get_hyperparams_rand, train_score_model


preds_file_index_cols = ['resample', 'cv_fold', 'model_number', 'data_split', 'task']

# Hardcode standard filenames:
train_val_test_split_filename = 'train_val_test_split.csv'
all_model_results_filename = 'GCNN_info_all_models.csv'
refit_model_results_filename = 'GCNN_info_refit_models.csv'
preds_filename = 'predictions.csv'
ext_preds_filename = None #'ext_predictions.csv'


# Allow some hyperparameters to have string values:
def eval_str(param):
    """
    Run eval() on string if possible, otherwise
    just return string.
    """
    try:
        out = eval(param)
    except NameError:
        out = param
    return out


def setup_run_training(run_input,
                       additional_params,
                       full_dataset,
                       ext_test_set={},
                       resample_n=0,
                       cv_n=None,
                       mod_i=0,
                       split_idxs={},
                       hp_dict={},
                       run_results={},
                       run_preds={},
                       preds_file='',
                       ext_preds_file='',
                      ):

    run_results[('model_info', 'resample_number')] = resample_n
    run_results[('model_info', 'cv_fold')] = cv_n
    run_results[('model_info', 'model_number')] = mod_i
    for hp_name, hp_val in hp_dict.items():
        run_results[('hyperparams', hp_name)] = hp_val

    # Save models and predictions:
    save_stage = ['all']
    if cv_n == 'refit':
        save_stage.append('refit')
    save_options = {}
    for opt in ['save_model', 'save_predictions']:
        # Make "refit" the default option if not specified:
        if run_input['training'].get(opt) is None:
            run_input['training'][opt] = 'refit'
        elif run_input['training'][opt] in save_stage:
            save_options[opt] = True
        else:
            save_options[opt] = False

    if save_options.get('save_model'):
        additional_params['model_dir'] = 'resample_{}_cv_{}_model_{}'.format(resample_n, cv_n, mod_i)
    if save_options.get('save_predictions'):
        if not os.path.isfile(preds_file):
            df_preds = pd.DataFrame(data=[],
                                    columns=preds_file_index_cols + \
                                            full_dataset.ids.tolist(),
                                    index=[])\
                         .to_csv(preds_file, index=False)
        if ('ext_datasets' in run_input) and \
           (len(run_input['ext_datasets']) > 0) and \
           not os.path.isfile(ext_preds_file):
            mol_idxs = [(set_name, molid) for set_name in run_input['ext_datasets']['_order'] 
                                          for molid in ext_test_set[set_name].ids]
            df_ext_preds = pd.DataFrame(data=[],
                                        columns=pd.MultiIndex.from_tuples(
                list(zip(*[preds_file_index_cols]*2)) + mol_idxs),
                                        index=[])\
                             .to_csv(ext_preds_file, index=False)

    # Save dataset copy and reshard:
    train_set = full_dataset.select(split_idxs['train'].to_list())
    if 'val' in split_idxs.index:
        val_set = full_dataset.select(split_idxs['val'].to_list())
    else:
        val_set = None
    if 'test' in split_idxs.index:
        test_set = full_dataset.select(split_idxs['test'].to_list())
    else:
        test_set = None

    # Check dataset split:
    check_dataset_split(*[dataset.ids for dataset in [train_set, val_set, test_set] 
                          if dataset is not None], 
                        n_samples=len(full_dataset.ids))

    # Submit model for training:
    run_results = \
    train_score_model(run_input=run_input,
                      hyperparams=hp_dict,
                      additional_params=additional_params,
                      train_set=train_set,
                      val_set=val_set,
                      test_set=test_set,
                      ext_test_set=ext_test_set,
                      run_results=run_results,
                      **save_options,
                      rand_seed=rand_seed,
                      preds_file=preds_filename,
                      ext_preds_file=ext_preds_filename)

    return run_results


# Read input details:
json_infile = sys.argv[1]
print('Reading input parameters from .json file: {}'.format(json_infile))
run_input = json.load(open(json_infile, 'r'))

# Choose metrics for classification or regression:
all_metrics = all_metrics[run_input['dataset']['mode']]

# If using a GraphConv model, limit TF threads to 1 (otherwise seems to cause problems on Archie), other 
# models seem to be alright and are able to run slightly faster when the number of threads is not set:
if 'GraphConv' in run_input['training']['model_fn_str']:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

# Generate empty Series object to store results:
run_results = setup_results_series(run_input, all_metrics)

# Fill in any info constant over all runs:
run_results[('training_info', 'deepchem_version')] = dc.__version__


# ==============================================
# Read any results from previous incomplete runs
# ==============================================

# File to save model details for each set of hyperparameters:
info_out = all_model_results_filename
if not os.path.isfile(all_model_results_filename):
    run_restart = False
    df_prev_runs = None
    info_out_header = True

# Read data from any previous runs:
else:
    run_restart = True
    df_prev = pd.read_csv(all_model_results_filename, sep=';', header=[0, 1])
    df_prev_runs = df_prev[[('model_info', 'resample_number'), 
                            ('model_info', 'cv_fold')] + \
                           [('hyperparams', hp_name) 
                            for hp_name in df_prev['hyperparams'].columns]]
    info_out_header = False

# Keep an empty copy to revert to after each run:
run_results_empty = run_results.copy()


# ===============
# Load dataset(s)
# ===============

print('Loading dataset: {}'.format(run_input['dataset']['dataset_file']))

# Load dataset:
load_data = GetDataset(**run_input['dataset'], 
                       **run_input['preprocessing'], 
                       **run_input['training'])
full_dataset, additional_params = load_data()

# Load external test sets if given:
ext_test_set = {}
if ('ext_datasets' in run_input) and len(run_input['ext_datasets']) > 0:
    for set_name in run_input['ext_datasets']['_order']:
        set_info = run_input['ext_datasets'][set_name]
        print('Loading external dataset: {} ({})'.format(set_info['dataset_file'], set_name))
        ext_test_set[set_name], _ = load_data(**set_info)

# DAG transformer if using DAG model:
reshard_size = None
if 'DAG' in run_input['training']['model_fn_str']:
    #reshard_size = 100
    max_atoms = max([mol.get_num_atoms() for mol in full_dataset.X])
    if ('ext_datasets' in run_input) and len(run_input['ext_datasets']) > 0:
        for set_name in run_input['ext_datasets']['_order']:
            max_atoms = max([mol.get_num_atoms() for mol in ext_test_set[set_name].X] + [max_atoms])
    additional_params['max_atoms'] = max_atoms
    transformer = dc.trans.DAGTransformer(max_atoms=max_atoms)
    full_dataset, _, _, ext_test_set, _ = \
    do_transforms(train_set=full_dataset,
                  ext_test_set=ext_test_set,
                  transformers=[transformer],
                  reshard_size=run_input['preprocessing']['reshard_size'])


# =============
# Split dataset
# =============

df_split_ids = nested_CV_splits(train_val_test_split_filename,
                                full_dataset.ids,
                                run_input,
                                run_restart=run_restart,
                                rand_seed=rand_seed)


# ===========================
# Setup hyperparameter tuning
# ===========================

# Exhaustive grid search:
if 'hyperparam_search' not in run_input['training'] or \
   run_input['training'].get('hyperparam_search') == 'grid':
    get_hyperparam_iter, n_hyperparams = get_hyperparams_grid(run_input['hyperparams'])

# Random search:
elif run_input['training']['hyperparam_search'] in ['rand', 'random']:
    get_hyperparam_iter, n_hyperparams = \
        get_hyperparams_rand(hyperparams=run_input['hyperparams'],
                             n_iter=run_input['training'].get('hyperparam_search_iterations'))

# GP:
elif run_input['training']['hyperparam_search'] == 'gp':
    raise NotImplementedError('GP not yet implemented')


# =======================
# Train individual models
# =======================

n_cpus = run_input['training'].get('n_cpus')

# Get number of available CPUs from slurm if run via slurm:
available_n_cpus = os.environ.get('SLURM_NTASKS')

# Check using the correct number of CPUs:
if available_n_cpus is not None:
    available_n_cpus = int(available_n_cpus)
    if (n_cpus is None) or ((n_cpus is not None) and (available_n_cpus != n_cpus)):
        if (n_cpus is not None) and (available_n_cpus != n_cpus):
            print('WARNING: {} CPUs requested, but {} available, will use {} CPUs available'.format(
                  n_cpus, available_cpus, available_cpus))
        n_cpus = available_n_cpus
        run_input['training']['n_cpus'] = available_n_cpus

# Add maxtasksperchild=1 to ensure that processes are only used once and so 
# memory is cleared after each model.  This is important for DeepChem 
# GraphConvModel models as they are not cleared from memory after training 
# (see: https://github.com/deepchem/deepchem/issues/1133), but this is 
# probably not required for other models.
pool = mp.Pool(n_cpus, maxtasksperchild=1)
pool_refit = mp.Pool(1, maxtasksperchild=1)
print('Training separate models on {} CPUs'.format(pool._processes))

model_results = []

# Add index number for selecting datasets:
df_split_ids.loc[full_dataset.ids, 'idx'] = range(len(full_dataset.ids))
df_split_ids['idx'] = df_split_ids['idx'].astype(int)

for resample_n, cv_n in df_split_ids.drop(columns=['idx']).columns:
    if cv_n == 'refit':
        continue

    split_idxs = df_split_ids.set_index((resample_n, cv_n))['idx']
    hyperparam_iter = get_hyperparam_iter()
    for mod_i, hp_dict in enumerate(hyperparam_iter):

        # Check whether set of hyperparameters have already been run:
        # THIS NEEDS MORE WORK!
        if (df_prev_runs is not None) and (len(df_prev_runs) > 0) and \
            np.any(
                np.all(
                    np.column_stack([(df_prev_runs[('model_info', 'resample_number')].astype(str) == str(resample_n)).to_list(),
                                     (df_prev_runs[('model_info', 'cv_fold')].astype(str) == str(cv_n))] + \
                                    [(df_prev_runs[('hyperparams', hp_name)].astype(str) == str(hp_val)).to_list()
                                     for hp_name, hp_val in hp_dict.items()]), 
                    axis=1)):
            continue

        print('Submitting resample: {}, cv fold: {}, hyperparameter combination: {}'.format(resample_n, cv_n, mod_i))

        model_results.append(
            pool.apply_async(setup_run_training,
                             kwds=(dict(run_input=run_input,
                                        additional_params=additional_params.copy(),
                                        full_dataset=full_dataset,
                                        resample_n=resample_n,
                                        cv_n=cv_n,
                                        mod_i=mod_i,
                                        split_idxs=split_idxs,
                                        hp_dict=hp_dict,
                                        run_results=run_results_empty.copy(),
                                        preds_file=preds_filename,
                                        ext_preds_file=ext_preds_filename))))


# ==================
# Refit final models
# ==================

# Fit best model for each resample:
df_cv_results = pd.DataFrame(columns=run_results.index)
df_final_results = pd.DataFrame(columns=run_results.index)
refit_models = []
refit_out_header = True
for model_result in model_results:
    df_model_result = model_result.get()
    resample_n, cv_n, mod_i = df_model_result['model_info'][['resample_number', 'cv_fold', 'model_number']]
    print('Finished training resample: {}, cv fold: {}, hyperparameter combination: {}'.format(resample_n, cv_n, mod_i))
    #df_model_result.to_csv(all_model_results_filename, index=False, mode='a', header=False)
    if info_out:
#        info_out.write(';'.join([str(i) for i in df_model_result.to_list()])+'\n')
#        info_out.flush()
        df_model_result.to_frame().T.to_csv(info_out, header=info_out_header, mode='a', index=False, sep=';')
        if info_out_header:
            info_out_header = False

    df_cv_results = df_cv_results.append(df_model_result, ignore_index=True)

    n_cv_splits = df_split_ids.loc[:,resample_n]\
                              .drop(columns='refit', errors='ignore')\
                              .shape[1]
    if df_cv_results.loc[df_cv_results[('model_info', 'resample_number')] == resample_n, 
                         [('model_info', 'cv_fold')]]\
                    .nunique()\
                    .squeeze() == n_cv_splits \
       and \
       np.all(df_cv_results.loc[df_cv_results[('model_info', 'resample_number')] == resample_n]\
                           .groupby(('model_info', 'cv_fold'))\
                           .count()[('model_info', 'model_number')] == n_hyperparams):

        selection_metric = run_input['training'].get('selection_metric')
        if selection_metric is None:
            selection_metric = 'loss'
        select_max_value = bool(run_input['training'].get('select_max_value'))

        hp_cols = [('hyperparams', hp_name) for hp_name in df_cv_results['hyperparams'].columns]
        best_mod_i_hp_idx = df_cv_results.loc[df_cv_results[('model_info', 'resample_number')] == resample_n]\
                                         .astype({hp_col : str for hp_col in hp_cols})\
                                         .groupby([('model_info', 'model_number')] + [('hyperparams', hp_name) for hp_name in df_cv_results['hyperparams'].columns])\
                                         .mean()\
                                         .sort_values(('val', selection_metric), ascending=not select_max_value)\
                                         .iloc[[0]]\
                                         .index

        best_mod_i = int(best_mod_i_hp_idx.get_level_values(('model_info', 'model_number')).values) #.squeeze()
        best_hp_idx = best_mod_i_hp_idx.droplevel([('model_info', 'model_number')])

        best_hp_dict = pd.Series(data=best_hp_idx.values.squeeze(),
                                 index=pd.MultiIndex.from_tuples(best_hp_idx.names))\
                         ['hyperparams']\
                         .apply(eval_str)\
                         .to_dict()

        # Average number of epochs if using early stopping:
        if run_input['training'].get('early_stopping') == True:
            run_input['training']['epochs'] = int(round(df_cv_results['training_info']['epochs'].mean(), 0))
            run_input['training']['early_stopping'] = False

        cv_n = 'refit'

        split_idxs = df_split_ids.set_index((resample_n, cv_n))['idx']
        print('Submitting resample: {}, cv fold: {}, hyperparameter combination: {}'.format(resample_n, cv_n, best_mod_i))

        # Probably doesn't need to be run as part of the pool, can be run immediately from the parent process, 
        # but launch within a new process in case of memory leaks:
        p = pool_refit.apply_async(setup_run_training,
                             kwds=(dict(run_input=run_input,
                                        additional_params=additional_params.copy(),
                                        full_dataset=full_dataset,
                                        resample_n=resample_n,
                                        cv_n=cv_n,
                                        mod_i=best_mod_i,
                                        split_idxs=split_idxs,
                                        hp_dict=best_hp_dict,
                                        run_results=run_results_empty.copy(),
                                        preds_file=preds_filename,
                                        ext_preds_file=ext_preds_filename)))

        df_model_result = p.get()
        df_model_result.to_frame().T.to_csv('GCNN_info_refit_models.csv', header=refit_out_header, mode='a', index=False, sep=';')
        refit_out_header = False
        resample_n, cv_n, mod_i = df_model_result['model_info'][['resample_number', 'cv_fold', 'model_number']]
        print('Finished training resample: {}, cv fold: {}, hyperparameter combination: {}'.format(resample_n, cv_n, mod_i))
        df_final_results = df_final_results.append(df_model_result, ignore_index=True)

pool.close()

pool_refit.close()


# =================================
# Get performance of ensemble model
# =================================

# Average refit predictions and then calculate performance:
if run_input["training"].get("calculate_ensemble_performance") and \
   run_input["training"].get("save_predictions") in ["refit", "all"]:
    print('Calculating performance of "ensemble" of refit models')
    df_preds = pd.read_csv(preds_filename, index_col=list(range(len(preds_file_index_cols))))

    df_av_preds = df_preds.loc[(df_preds.index.get_level_values('cv_fold') == 'refit') & \
                               (df_preds.index.get_level_values('data_split') == 'test')]\
                          .reset_index(level='task')\
                          .groupby('task')\
                          .mean()

    df_av_preds.index = pd.MultiIndex.from_tuples([(-1, 'av_over_refits', -1, 'test',
                                                    task) for task in df_av_preds.index])

    # Save average predictions to file:
    df_av_preds.to_csv(preds_filename, mode='a', header=False)

    run_results = run_results_empty.copy()

    df_av_preds.dropna(axis=1, inplace=True)

    train_vals = pd.read_csv(run_input["dataset"]["dataset_file"])\
                   .set_index(run_input["dataset"]["id_field"])\
                   .loc[df_av_preds.columns, run_input["dataset"]["tasks"][0]]\
                   .to_numpy()

    for metric_name in all_metrics['_order']:
        metric = all_metrics[metric_name]
        run_results[('test', metric.name)] = round(
                                metric.compute_metric(train_vals,
                                                      df_av_preds.loc[(-1, 'av_over_refits', -1, 'test', run_input["dataset"]["tasks"][0])].to_numpy(),
                                                     ), 3)

    # Save stats to file:
    # Maintain header order:
    df_ens_model_result = pd.read_csv('GCNN_info_refit_models.csv', sep=';', header=[0, 1], nrows=0)
    df_ens_model_result = df_ens_model_result.append(run_results.to_frame()\
                                                                .T)\
                                             [df_ens_model_result.columns]
    df_ens_model_result.to_csv('GCNN_info_refit_models.csv', header=False, mode='a', index=False, sep=';')


# =======================
# Get average performance
# =======================

# Get average, standard deviation and standard error:
df_av_results = \
get_average_model_performance(df_final_results, mode=run_input['dataset']['mode'])

df_av_results.to_csv(run_input['training']['model_fn_str'].split('.')[-1]+'_info_av_performance.csv')

print('Average performance on test set:')
print(df_av_results.loc['test'].to_string(index=True))
