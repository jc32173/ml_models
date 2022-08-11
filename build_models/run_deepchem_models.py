
# Use logger rather then print

#from tensorflow.keras import backend as K
import tensorflow as tf
#from tf.keras import backend as K
# Works a bit:
#from keras import backend as K

from multiprocessing import Process
from rdkit import Chem
import deepchem as dc
import pickle as pk
import pandas as pd
import numpy as np
import math
import sys, os
from datetime import datetime
import json
from itertools import product


# Set random seeds to make results fully reproducable:
if len(sys.argv) >= 3:
    rand_seed = int(sys.argv[2])
    np.random.seed(rand_seed+123)
    tf.random.set_seed(rand_seed+456)
else:
    rand_seed = None

# Update this when code is moved to programs folder:
sys.path.insert(0, '/users/xpb20111/programs/deepchem')

# Define deepchem metrics:
from build_models.dc_metrics import all_metrics
from build_models.dc_preprocess import GetDataset, train_val_test_split, mol_desc_scaling_selection
from build_models.dc_training import get_hyperparams_grid, train_model

# Read input details:
json_infile = sys.argv[1]
print('Reading input parameters from .json file: {}'.format(json_infile))
run_input = json.load(open(json_infile, 'r'))

# If GraphConv model limit TF threads to 1 (otherwise seems to cause problems on Archie), other models
# seem to be alright and are able to run slightly faster when the number of threads is not set:
if 'GraphConv' in run_input['training']['model_fn_str']:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)

# Use multiindex header:
header = []

# General information, constant over all runs:
for level1 in ['model_info']:
    for level2 in ['model_number']: #, 'rand_seed']: # 'dataset', 'split', 'model', 'np_seed', 'tf_seed', 'split_seed']:
        header.append((level1, level2))

# Stats for each run:
if 'ext_datasets' in run_input:
    ext_dataset_names = run_input['ext_datasets']['_order']
else:
    ext_dataset_names = []
for level1 in ['train', 'val', 'test'] + ext_dataset_names:
    for level2 in all_metrics['_order'] + ['y_stddev'] + ['loss']:
        header.append((level1, level2))
    if run_input['training'].get('uncertainty'):
        header.append((level1, 'uncertainty-error_pearson'))
        header.append((level1, 'uncertainty-error_spearman'))
        header.append((level1, 'uncertainty-error_kendall'))

# Training info for each run:
for level1 in ['training_info']:
    for level2 in ['date', 'training_time', 'results_epoch', 'best_epoch', 'deepchem_version']:
        header.append((level1, level2))

# Hyper parameters for each run:
for level1 in ['hyperparams']:
    for level2 in run_input['hyperparams']['_order']:
        header.append((level1, level2))

run_results = pd.Series(index=pd.MultiIndex.from_tuples(header))
run_results[('training_info', 'deepchem_version')] = dc.__version__

## Fill in any info constant for all runs:
#run_results'model_number', 'dataset', 'split', 'model', 'np_seed', 'tf_seed', 'split_seed' 
s_ls = []
for level1 in ['model_info', 'dataset', 'preprocessing', 'splitting', 'feature_selection', 'training']:
    if level1 not in run_input.keys():
        continue
    s = pd.Series(data=run_input[level1])
    s.index = pd.MultiIndex.from_tuples([(level1, level2) for level2 in s.index])
    s_ls.append(s)
run_results = pd.concat([run_results] + s_ls, verify_integrity=True)

# Use environment variables to get N CPUs/nodes and slurm job ID if run on slurm:
# See: https://hpcc.umd.edu/hpcc/help/slurmenv.html
run_results[('training_info', 'n_cpus_nodes')] = '['+str(os.environ.get('SLURM_CPUS_ON_NODE'))+']'
run_results[('training_info', 'slurm_jobid')] = os.environ.get('SLURM_JOB_ID')

# Add in extra columns here, eventually make this not hard coded:
run_results[('training_info', 'n_atom_feat')] = np.nan
if 'Weave' in run_input['training']['model_fn_str']:
    run_results[('training_info', 'n_pair_feat')] = np.nan
if 'feature_selection' in run_input.keys() and \
   run_input['feature_selection'].get('selection_method') == 'PCA':
    run_results[('training_info', 'N_PCA_feats')] = np.nan

# File to save model details for each set of hyperparameters:
if not os.path.isfile('GCNN_info_all_models.csv'):
    df_prev_hp = None
    info_out = open('GCNN_info_all_models.csv', 'w')
    info_out.write(';'.join(run_results.index.get_level_values(0))+'\n')
    info_out.write(';'.join(run_results.index.get_level_values(1))+'\n')
    # Need to flush here to ensure header is only written once and removed from buffer, otherwise header is written before the results from each model:
    info_out.flush()
    mod_i = 0

# Read data from any previous runs:
else:
    df_prev = pd.read_csv('GCNN_info_all_models.csv', sep=';', header=[0, 1])
    df_prev_hp = df_prev['hyperparams']
    if len(set(df_prev.columns) & set(run_results.index)) != len(df_prev.columns):
        sys.exit('ERROR')
    mod_i = df_prev[('model_info', 'model_number')].max() + 1
    info_out = open('GCNN_info_all_models.csv', 'a')
    run_results = run_results[df_prev.columns]

# Keep an empty copy to revert to after each run:
run_results_empty = run_results.copy()

print('Loading dataset: {}'.format(run_input['dataset']['dataset_file']))
# Load dataset:
DAGModel = False
if 'DAG' in run_input['training']['model_fn_str']:
    DAGModel = True
load_data = GetDataset(**run_input['dataset'], 
                       **run_input['preprocessing'],
                       DAGModel=DAGModel)
dataset, additional_params, transformers = load_data()

#if 'output_types' in run_input.keys():
#    additional_params['output_types'] = run_input['output_types']

print('Splitting dataset')
# Split dataset:
train_set, val_set, test_set, transformers = \
    train_val_test_split(dataset,
                         **run_input['splitting'],
                         dataset_file=run_input['dataset']['dataset_file'],
                         transformer='norm',
                         transformers=transformers,
                         rand_seed=rand_seed)

# Load external test sets if given:
ext_test_set = {}
if ('ext_datasets' in run_input) and len(run_input['ext_datasets']) > 0:
    for set_name in run_input['ext_datasets']['_order']:
        set_info = run_input['ext_datasets'][set_name]
        print('Loading external dataset: {} ({})'.format(set_info['dataset_file'], set_name))
        ext_test_set[set_name], _, _ = load_data(**set_info)
        # Need to run transformers too:
        # NEEDS ATTENTION!!:
        # Better to do transforms later?  Can then also include max_atoms
        ext_test_set[set_name] = transformers[-1].transform(ext_test_set[set_name])

if 'DAG' in run_input['training']['model_fn_str']:
    reshard_size = 100
    max_atoms = max([mol.get_num_atoms() for mol in dataset.X])
    if ('ext_datasets' in run_input) and len(run_input['ext_datasets']) > 0:
        for set_name in run_input['ext_datasets']['_order']:
            max_atoms = max([mol.get_num_atoms() for mol in ext_test_set[set_name].X] + [max_atoms])
    additional_params['max_atoms'] = max_atoms
    transformer = dc.trans.DAGTransformer(max_atoms=max_atoms)
    train_set = transformer.transform(train_set)
    train_set.reshard(reshard_size)
    val_set = transformer.transform(val_set)
    val_set.reshard(reshard_size)
    if test_set:
        test_set = transformer.transform(test_set)
        test_set.reshard(reshard_size)
    if ('ext_datasets' in run_input) and len(run_input['ext_datasets']) > 0:
        for set_name in run_input['ext_datasets']['_order']:
            ext_test_set[set_name] = transformer.transform(ext_test_set[set_name])
            ext_test_set[set_name].reshard(reshard_size)
    transformers.append(transformer)

#print()
# Select training set descriptors?  Maybe use own transformer?:
# ...
#desc = calc_rdkit_desc()

# Scale any molecular descriptors and optionally 
# use PCA/PLS to select descriptors:
# ALSO NEED TO OUTPUT AN ASSOCIATED TRANSFORMER:
if 'feature_selection' in run_input:
#if True:
# As a function:
#    train_set, other_datasets = \
#        mol_desc_scaling_selection(train_set, 
#                                   other_datasets=[val_set, test_set] + [ext_test_set[set_name] for set_name in run_input['ext_datasets']['_order']], 
#                                   **run_input['feature_selection'], 
#                                   training_info=additional_params)
#    # Unpack other datasets:
#    val_set, test_set = other_datasets[:2]
#    for set_i, set_name in enumerate(run_input['ext_datasets']['_order']):
#        ext_test_set[set_name] = other_datasets[set_i+2]

    #train_set, feat_transformer = \
    feat_transformer = \
    mol_desc_scaling_selection(dataset=train_set, 
                               **run_input['feature_selection'], 
                               training_info=additional_params)
    train_set = feat_transformer.transform(train_set)
    val_set = feat_transformer.transform(val_set)
    if test_set:
        test_set = feat_transformer.transform(test_set)
    if ('ext_datasets' in run_input) and len(run_input['ext_datasets']) > 0:
        for set_name in run_input['ext_datasets']['_order']:
            ext_test_set[set_name] = feat_transformer.transform(ext_test_set[set_name])
    transformers.append(feat_transformer)

# Hyperparameter tuning:
print('Selecting hyperparameters:')

# Exhaustive grid search:
if 'hyperparam_search' not in run_input['training'] or \
   run_input['training'].get('hyperparam_search') == 'grid':
    hyperparam_iter = get_hyperparams_grid(run_input['hyperparams'],
                                           df_prev_hp=df_prev_hp)
# Random search:
elif run_input['training']['hyperparam_search'] == 'rand':
    hyperparam_iter = \
        get_hyperparams_rand(hyperparams=run_input['hyperparams'],
                             n_iter=run_input['hyperparam_search_iterations'],
                             df_prev_hp=df_prev_hp)

for hp_dict, df_hp in hyperparam_iter:
    for hp in df_hp.columns:
        hp_val = df_hp[hp].iloc[0]
        run_results[('hyperparams', hp)] = hp_val
        print('    {:10}: {}'.format(hp, hp_val))

    additional_params['model_dir'] = 'trained_model'+str(mod_i)

    # Run training in a separate process by default to match previous 
    # behaviour and prevent memory leak:
    if run_input['training'].get('separate_process') is None or \
       run_input['training']['separate_process'] == True:

        print("Running training in separate process to prevent memory leak", flush=True)
        p = Process(target=train_model, kwargs=(dict(mod_i=mod_i,
                                                     **run_input['training'],
                                                     hyperparams=hp_dict,
                                                     additional_params=additional_params,
                                                     train_set=train_set,
                                                     val_set=val_set,
                                                     test_set=test_set,
                                                     ext_test_set=ext_test_set,
                                                     transformers=transformers,
                                                     out_file=info_out,
                                                     run_results=run_results,
                                                     rand_seed=rand_seed)))
        p.start()
        p.join()

    else:
        print("Running training without specifying a new process", flush=True)
        train_model(mod_i=mod_i,
                    **run_input['training'],
                    hyperparams=hp_dict,
                    additional_params=additional_params,
                    train_set=train_set,
                    val_set=val_set,
                    test_set=test_set,
                    ext_test_set=ext_test_set,
                    transformers=transformers,
                    out_file=info_out,
                    run_results=run_results,
                    rand_seed=rand_seed)

    # Increment model counter:
    mod_i += 1

    run_results = run_results_empty.copy()

info_out.close()
