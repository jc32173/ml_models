from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from rdkit import Chem
import deepchem as dc
import pandas as pd
import numpy as np
import sys

#sys.path.insert(0, '/users/xpb20111/programs/deepchem')


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

    if split_stage == 'outer':
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


def train_test_split(data_ids,
                     split_method='random',
                     dataset_file=None,
                     strat_field=None,
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
        splitter = StratifiedShuffleSplit(1, 
                                          test_size=1-frac_train, 
                                          random_state=rand_seed
                                         )
        data_splits = splitter.split(data_ids, c)

    elif split_method == 'predefined_lipo':

        # Split using predefined split used in MoleculeNet benchmark:

        train_set_ids, test_set_ids = get_lipo_split_ids(split_stage)
        yield train_set_ids, test_set_ids

    # Return the IDs from each split:
    for train_set_idxs, test_set_idxs in data_splits:
        train_set_ids = data_ids[train_set_idxs]
        test_set_ids = data_ids[test_set_idxs]
        yield train_set_ids, test_set_ids
