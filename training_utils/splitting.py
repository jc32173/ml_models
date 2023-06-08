from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, GroupShuffleSplit, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
import deepchem as dc
import pandas as pd
import numpy as np
import sys
import os


# Check no overlap between data sets:
def check_dataset_split(train_set_ids=[],
                        val_set_ids=[],
                        test_set_ids=[],
                        n_samples=False):
    """
    Function to check that there is no overlap between train, 
    validation and test datasets and all data is accounted for.
    """

    if (set(train_set_ids) & set(val_set_ids)) | \
       (set(train_set_ids) & set(test_set_ids)) | \
       (set(val_set_ids) & set(test_set_ids)):
        raise ValueError('Overlap between train, validation and test sets.')
    if n_samples and \
        len(set(train_set_ids) | set(val_set_ids) | set(test_set_ids)) != n_samples:
        print(n_samples)
        raise ValueError('Some data not assigned to train, validation or test sets.')


def apply_lipo_split_to_dataset(dataset):
    """
    Split the MoleculeNet logD dataset into the same train/validation/test
    sets as used in the MoleculeNet benchmark.  Used to compare the performance 
    of different featurizers to the benchmark.
    """

    tasks, datasets, transformers = dc.molnet.load_lipo(featurizer='GraphConv')
    train_set, val_set, test_set = datasets
    train_ids = train_set.ids
    test_ids = test_set.ids
    val_ids = val_set.ids

    train_set = dataset.select([i for i in range(len(dataset)) 
                                    if dataset.ids[i] in train_ids])
    test_set = dataset.select([i for i in range(len(dataset)) 
                                   if dataset.ids[i] in test_ids])
    val_set = dataset.select([i for i in range(len(dataset)) 
                                  if dataset.ids[i] in val_ids])

    check_dataset_split(train_set.ids, val_set.ids, test_split.ids)

    return train_set, val_set, test_split


def get_lipo_split_ids(split_stage):
    """
    Split the MoleculeNet logD dataset into the same train/validation/test
    sets as used in the MoleculeNet benchmark.  Used to compare the performance 
    of different featurizers to the benchmark.
    """

    tasks, datasets, transformers = dc.molnet.load_lipo(featurizer='GraphConv')
    train_set, val_set, test_set = datasets
    train_set_ids = train_set.ids
    test_set_ids = test_set.ids
    val_set_ids = val_set.ids

    check_dataset_split(train_set_ids, val_set_ids, test_split_ids)

    if split_stage == 'train_test_split':
        yield np.concatenate([train_set_ids, val_set_ids]), test_set_ids
    else:
        yield train_set_ids, val_set_ids


#def train_val_test_split(dataset,
#                         split_method='random',
#                         dataset_file=None,
#                         mode='regression',
#                         strat_field=None,
#                         frac_train=0.7,
#                         frac_valid=0.15,
#                         frac_test=0.15,
#                         rand_seed=None,
#                         norm_transform=True,
#                         reshard_size=False):
#    """
#    Function to split a dataset into train, validation and test sets 
#    using different methods.
#    """
#
#    if split_method in ['rand', 'random']:
#
#        # Random split using deepchem:
#
#        splitter = dc.splits.RandomSplitter()
#
#        split_n = 0
#        while True:
#            train_set, val_set, test_set = \
#                splitter.train_valid_test_split(dataset,
#                                                frac_train=frac_train,
#                                                frac_valid=frac_valid,
#                                                frac_test=frac_test,
#                                                seed=rand_seed+split_n)
#            split_n += 1
#            reshard(train_set, val_set, test_set, reshard_size)
#            yield train_set, val_set, test_set
#
#    elif split_method == 'k-fold':
#
#        # k-fold split using sklearn:
#
#        splitter = KFold(n_splits=k_fold, 
#                         shuffle=True, 
#                         random_state=rand_seed)
#
#        for train_idx, val_idx in splitter.split(dataset.X):
#            #train_sets.append(dataset.select(train_idx))
#            #val_sets.append(dataset.select(val_idx))
#        #test_set = None
#            train_set = dataset.select(train_idx)
#            val_set = dataset.select(val_idx)
#            test_set = None
#            reshard(train_set, val_set, test_set, reshard_size)
#            yield train_set, val_set, test_set
#
#    elif split_method in ['fp', 'fingerprint']:
#
#        # Split based on fingerprint similarity:
#
#        splitter = dc.splits.FingerprintSplitter()
#        train_set, val_set, test_set = \
#            splitter.train_valid_test_split(dataset,
#                                            frac_train=frac_train,
#                                            frac_valid=frac_valid,
#                                            frac_test=frac_test,
#                                            seed=rand_seed)
#    elif split_method == 'butina':
#
#        # Split based on butina clustering:
#
#        splitter = dc.splits.ButinaSplitter(cutoff=0.6)
#        train_set, val_set, test_set = \
#            splitter.train_valid_test_split(dataset,
#                                            frac_train=frac_train,
#                                            frac_valid=frac_valid,
#                                            frac_test=frac_test,
#                                            seed=rand_seed)
#
#    elif split_method in ['strat', 'stratified']:
#
#        # Stratified split using sklearn:
#
#        df = pd.read_csv(dataset_file)
#        # If canon_SMILES is not index:
#        # df.set_index(id_field, inplace=True, verify_integrity=True)
#        c = df[strat_field]
#
#        # Do two separate splits to get three sets:
#        splitter = StratifiedShuffleSplit(1, 
#                                          test_size=1-frac_train, 
#                                          random_state=rand_seed
#                                         ).split(dataset.X, c)
#        train_idx, val_test_idx = list(splitter)[0]
#
#        train_set = dataset.select(train_idx)
#        val_test_set = dataset.select(val_test_idx)
#
#        # Have to split val_test_set in second split to ensure no overlap:
#        splitter = StratifiedShuffleSplit(1, 
#                                          test_size=frac_test/(frac_valid+frac_test), 
#                                          random_state=rand_seed+100
#                                         ).split(val_test_set.X, c[val_test_idx])
#        val_idx, test_idx = list(splitter)[0]
#
#        val_set = val_test_set.select(val_idx)
#        test_set = val_test_set.select(test_idx)
#
#    elif split_method == 'predefined_lipo':
#
#        # Split using predefined split used in MoleculeNet benchmark:
#
#        train_set, val_set, test_set = get_lipo_split(dataset)
#        reshard(train_set, val_set, test_set, reshard_size)
#        yield train_set, val_set, test_set
#
#    elif split_method == 'predefined':
#
#        # Split using predefined split based on separate column:
#
#        df = pd.read_csv(dataset_file)
#        c = df[strat_field]
#
#        train_set = dataset.select(df.loc[c == 'train'].index)
#        val_set = dataset.select(df.loc[c == 'val'].index)
#        test_set = dataset.select(df.loc[c == 'test'].index)
#        reshard(train_set, val_set, test_set, reshard_size)
#        yield train_set, val_set, test_set


def train_test_split(data_ids=[],
                     split_method='random',
                     dataset_file=None,
                     strat_field=None,
                     id_field=None,
                     n_splits=5,
                     split_stage=None,
                     frac_train=0.7,
                     frac_test=0.3,
                     rand_seed=None):
    """
    Function to split a dataset into train and test sets 
    using different methods.
    """

    if split_method in ['rand', 'random']:

        # Random split:

        splitter = ShuffleSplit(n_splits=n_splits,
                                train_size=frac_train,
                                test_size=frac_test,
                                random_state=rand_seed)
        data_splits = splitter.split(data_ids)

    elif split_method == 'k-fold':

        # k-fold split using sklearn:

        splitter = KFold(n_splits=n_splits,
                         shuffle=True,
                         random_state=rand_seed)
        data_splits = splitter.split(data_ids)

#    elif split_method in ['fp', 'fingerprint']:
#
#        # Split based on fingerprint similarity:
#
#        splitter = dc.splits.FingerprintSplitter()
#        train_set, val_set, test_set = \
#            splitter.train_valid_test_split(dataset,
#                                            frac_train=frac_train,
#                                            frac_valid=frac_valid,
#                                            frac_test=frac_test,
#                                            seed=rand_seed)
#    elif split_method == 'butina':
#
#        # Split based on butina clustering:
#
#        splitter = dc.splits.ButinaSplitter(cutoff=0.6)
#        data_split = \
#            splitter.train_valid_test_split(dataset,
#                                            frac_train=frac_train,
#                                            frac_valid=frac_valid,
#                                            frac_test=frac_test,
#                                            seed=rand_seed)

    elif split_method in ['strat', 'stratified']:

        # Stratified split using sklearn:

        df = pd.read_csv(dataset_file)
        # If canon_SMILES is not index:
        # df.set_index(id_field, inplace=True, verify_integrity=True)
        c = df[strat_field]

        # Do two separate splits to get three sets:
        splitter = StratifiedShuffleSplit(n_splits=n_splits, 
                                          test_size=1-frac_train, 
                                          random_state=rand_seed
                                         )
        data_splits = splitter.split(data_ids, c)

#    elif split_method == 'murcko':
#
#        # Split based on Murko scaffolds:
#
#        df = pd.read_csv(dataset_file).set_index(id_field)
#        df['scaffolds'] = \
#            [MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(smi), 
#                                  includeChirality=False)
#             for smi in df[strat_field]]
#        c = df['scaffolds'].drop_duplicates()\
#                           .to_numpy()
#
#        splitter = ShuffleSplit(n_splits=n_splits,
#                                test_size=1-frac_train,
#                                random_state=rand_seed
#                               )
#        #data_splits = splitter.split(data_ids, c)
#        data_splits = splitter.split(c)
#
#        for train_set_idxs, test_set_idxs in data_splits:
#            train_set_ids = df.loc[df['scaffolds'].isin(c[train_set_idxs])].index
#            test_set_ids = df.loc[df['scaffolds'].isin(c[test_set_idxs])].index
#            yield train_set_ids, test_set_ids

    elif (split_method == 'murcko') or \
         (split_method == 'murcko_k-fold'):

        # Set up group split based on Murko scaffolds:

        df = pd.read_csv(dataset_file).set_index(id_field)
        df = df.loc[data_ids]
        df['scaffolds'] = \
            [MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(smi), 
                                  includeChirality=False)
             for smi in df[strat_field]]
        #c = df['scaffolds'].drop_duplicates()\
        #                   .to_numpy()

        if split_method == 'murcko':
            splitter = GroupShuffleSplit(n_splits=n_splits,
                                         test_size=1-frac_train,
                                         random_state=rand_seed
                                        )

        elif split_method == 'murcko_k-fold':
            splitter = GroupKFold(n_splits=n_splits,
                                 )

        #data_splits = splitter.split(data_ids, c)
        data_splits = splitter.split(data_ids, groups=df['scaffolds'])

#    elif split_method == 'murcko_rebalanced':
#
#        # Set up group split based on Murko scaffolds, then rebalance 
#        # to get roughly the right train:test ratio:
#
#        df = pd.read_csv(dataset_file).set_index(id_field)
#        df['scaffolds'] = \
#            [MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(smi), 
#                                  includeChirality=False)
#             for smi in df[strat_field]]
#        c = df['scaffolds'].drop_duplicates()\
#                           .to_numpy()
#
#        splitter = GroupShuffleSplit(n_splits=n_splits,
#                                     test_size=1-frac_train,
#                                     random_state=rand_seed
#                                    )
#        data_splits = splitter.split(data_ids, c)
#        #data_splits = splitter.split(c)
#
#        for train_set_idxs, test_set_idxs in data_splits:
#            train_set_ids = data_ids[train_set_idxs]
#            test_set_ids = data_ids[test_set_idxs]
#
#            actual_frac_train = len(train_set_ids)/len(data_ids)
#            print(actual_frac_train)
#
#            # Rebalance if necessary:
#            if np.abs(frac_train - actual_frac_train) > imbalance_tolerance:
#                df_curr_split = df['scaffold']
#                df_curr_split.loc[train_set_ids, 'set'] = 'train'
#                df_curr_split.loc[test_set_ids, 'set'] = 'test'
#
#                df_curr_split = \
#                df_curr_split.groupby(['scaffold', 'set'])\
#                             .agg(group_ids=('ID' : lambda i: list(i)), 
#                                  group_size=('scaffold', 'count'))
#
#                imbalance_tolerance = 0.05
#                while np.abs(frac_train - len(df_curr_split.loc[df_curr_split['set'] == 'train'])) < imbalance_tolerance:
#                    if actual_frac_train > frac_train:
#                        frac_train - imbalance_tolerance
#                        df_curr_split.loc[df_curr_split['group_size'] < ]
#                        #...
#
#            n_imbalance = round(frac_train*len(data_ids)) - len(train_set_idxs)
#            if n_imbalance
#
#            yield train_set_ids, test_set_ids

    elif split_method == 'predefined_lipo':

        # Split using predefined split used in MoleculeNet benchmark:

        train_set_ids, test_set_ids = get_lipo_split_ids(split_stage)
        yield train_set_ids, test_set_ids

    # Return the IDs from each split:
    for train_set_idxs, test_set_idxs in data_splits:
        train_set_ids = data_ids[train_set_idxs]
        test_set_ids = data_ids[test_set_idxs]
        yield train_set_ids, test_set_ids


def nested_CV_splits(train_val_test_split_filename,
                     dataset_ids,
                     run_input,
                     rand_seed=None):
    """
    Produce train/validation/test splits for nested CV.
    """

    # Ensure dataset_ids are a numpy array:
    if isinstance(dataset_ids, pd.core.frame.DataFrame) or \
       isinstance(dataset_ids, pd.core.series.Series):
        dataset_ids = dataset_ids.squeeze().to_numpy()
    elif isinstance(dataset_ids, list):
        dataset_ids = np.array(dataset_ids)

#    # If previous train/val/test splits have been saved use these:
#    if os.path.isfile(train_val_test_split_filename):
#        print('Reading dataset splits from:', train_val_test_split_filename)
#        df_split_ids = pd.read_csv(train_val_test_split_filename, header=[0, 1], index_col=0)
#        #df_split_ids.set_index('ID', verify_integrity=True, inplace=True)

    if run_input['train_test_split'].get('from_file') is not None:
        train_test_split_filename = run_input['train_test_split']['from_file']
        if not os.path.isfile(train_test_split_filename):
            raise ValueError('')

        print('Reading dataset splits from:', train_test_split_filename)
        df_split_ids = pd.read_csv(train_val_test_split_filename, header=[0, 1], index_col=0)

        if run_input['train_test_split'].get('from_file') == True:
            return df_split_ids

        else:
            def get_train_test_split_iter(df):
                for resample_n in df.columns:
                    yield df.index[df[resample_n] != 'train'], df.index[df[resample_n] == 'train']
            train_test_split_iter = get_train_test_split_iter(df_split_ids)

    else:
        print('Generating dataset splits and saving to:', train_val_test_split_filename)
        df_split_ids = pd.DataFrame(data=[],
                                    index=dataset_ids,
                                    columns=pd.MultiIndex.from_tuples([], names=['resample', 'cv']))
        # Make first level of column names strings:
        #print(df_split_ids.columns)
        #df_split_ids.columns = \
        #    df_split_ids.columns.set_levels(df_split_ids.columns\
        #                                                .levels[0]\
        #                                                .astype(str), 
        #                                    level=0)
        df_split_ids.index.rename('ID', inplace=True)

        train_test_split_iter = train_test_split(dataset_ids,
                                                 **run_input['train_test_split'],
                                                 dataset_file=run_input['dataset']['dataset_file'],
                                                 id_field=run_input['dataset'].get('id_field'),
                                                 rand_seed=rand_seed)

    for resample_n, [train_val_ids, test_set_ids] in enumerate(train_test_split_iter):
        resample_n = str(resample_n)

        train_val_split_iter = train_test_split(train_val_ids,
                                            **run_input['train_val_split'],
                                            dataset_file=run_input['dataset']['dataset_file'],
                                            id_field=run_input['dataset'].get('id_field'),
                                            rand_seed=rand_seed)

        for cv_n, [train_set_ids, val_set_ids] in enumerate(train_val_split_iter):
            cv_n = str(cv_n)

            check_dataset_split(train_set_ids,
                                val_set_ids,
                                test_set_ids,
                                n_samples=len(dataset_ids))

            df_split_ids[(resample_n,
                          cv_n)] = np.nan
            df_split_ids[(resample_n,
                          cv_n)].loc[train_set_ids] = 'train'
            df_split_ids[(resample_n,
                          cv_n)].loc[test_set_ids] = 'test'
            df_split_ids[(resample_n,
                          cv_n)].loc[val_set_ids] = 'val'

        df_split_ids[(resample_n,
                      'refit')] = np.nan
        df_split_ids[(resample_n,
                      'refit')].loc[set(train_set_ids) | set(val_set_ids)] = 'train'
        df_split_ids[(resample_n,
                      'refit')].loc[test_set_ids] = 'test'

    # Save split assignments to file:
    df_split_ids.to_csv(train_val_test_split_filename)

    return df_split_ids
