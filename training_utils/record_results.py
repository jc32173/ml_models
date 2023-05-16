import numpy as np
import pandas as pd
import os

def setup_results_series(run_input, all_metrics):
    """
    Set up empty Series object to store results.
    """

    # Use multiindex header:
    header = []

    # Model number:
    for level1 in ['model_info']:
        for level2 in ['resample_number', 'cv_fold', 'model_number']: #, 'rand_seed']: # 'dataset', 'split', 'model', 'np_seed', 'tf_seed', 'split_seed']:
            header.append((level1, level2))

    # Stats for each run:
    ext_dataset_names = []
    if 'ext_datasets' in run_input:
        ext_dataset_names = run_input['ext_datasets']['_order']
    level2_metrics = []
    if run_input['dataset']['mode'] == 'regression':
        level2_metrics = ['y_stddev']
    for level1 in ['train', 'val', 'test'] + ext_dataset_names:
        for level2 in all_metrics['_order'] + level2_metrics + ['loss']:
        #for level2 in [m.name for m in all_metrics] + level2_metrics + ['loss']:
            header.append((level1, level2))

    # Training info for each run:
    for level1 in ['training_info']:
        for level2 in ['date', 'training_time', 'epochs', 'deepchem_version']:
            header.append((level1, level2))

    # Hyperparameters for each run:
    for level1 in ['hyperparams']:
        for level2 in run_input['hyperparams']['_order']:
            header.append((level1, level2))

    # Set up empty Series:
    run_results = pd.Series(index=pd.MultiIndex.from_tuples(header))

    # Fill in info from run_input:
    s_ls = []
    for level1 in ['model_info', 'dataset', 'preprocessing', 'train_test_split', 'train_val_split', 'feature_selection', 'training']:
        if level1 not in run_input.keys():
            continue
        s = pd.Series(data=run_input[level1])
        s.index = pd.MultiIndex.from_tuples([(level1, level2) for level2 in s.index])
        s_ls.append(s)
    run_results = pd.concat([run_results] + s_ls, verify_integrity=True)

    # Use environment variables to get number of CPUs/nodes and slurm job ID if run on slurm:
    # See: https://hpcc.umd.edu/hpcc/help/slurmenv.html
    run_results[('training_info', 'n_cpus_nodes')] = '['+str(os.environ.get('SLURM_CPUS_ON_NODE'))+']'
    run_results[('training_info', 'slurm_jobid')] = os.environ.get('SLURM_JOB_ID')

    # Extra columns - this needs altering:
    run_results[('training_info', 'n_atom_feat')] = np.nan
    if run_input['dataset']['mode'] == 'classification':
        run_results[('training_info', 'n_classes')] = np.nan
    if 'Weave' in run_input['training']['model_fn_str']:
        run_results[('training_info', 'n_pair_feat')] = np.nan
    if 'feature_selection' in run_input.keys() and \
       run_input['feature_selection'].get('selection_method') == 'PCA':
        run_results[('training_info', 'N_PCA_feats')] = np.nan

    return run_results

