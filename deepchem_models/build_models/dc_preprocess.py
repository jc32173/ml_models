from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from rdkit import Chem
import deepchem as dc
import pandas as pd
import numpy as np
import math
import sys

#sys.path.insert(0, '/users/xpb20111/programs/deepchem')

from deepchem_models.modified_deepchem.PreprocessFeaturizerWrapper import PreprocessFeaturizerWrapper
#from modified_deepchem.WeaveFeaturizerWrapper import WeaveFeaturizerWrapper
from deepchem_models.modified_deepchem.FeaturizerOptDesc import ConvMolFeaturizer_OptDesc

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


class GetDataset():
    """
    Class to load dataset, including preprocessing SMILES and featurizing.
    """

    def __init__(self,
                 dataset_file='', 
                 tasks=[''],
                 feature_field='',
                 id_field='',
                 featurizer=None,
                 mode='regression',
                 uncertainty=False,
                 tauto=False,
                 ph=None,
                 phmodel='OpenEye',
                 rdkit_desc=False,
                 extra_desc=[],
                 **kwargs):

        self.dataset_file = dataset_file
        if isinstance(tasks, str):
            tasks = [tasks]
        self.tasks = tasks

        if isinstance(extra_desc, str):
            extra_desc = [extra_desc]

        self.additional_params = {'n_tasks' : len(tasks), 
                                  'mode' : mode, 
                                  'uncertainty' : uncertainty}

        self.feature_field=feature_field
        self.id_field=id_field

        if isinstance(featurizer, list) or tauto or ph or rdkit_desc or extra_desc:
            print('Using PreprocessFeaturizerWrapper')
            self.featurizer = PreprocessFeaturizerWrapper(
                     smiles_column=feature_field,
                     featurizer=[eval(f) for f in featurizer],
                     tauto=tauto,
                     ph=ph,
                     phmodel=phmodel,
                     rdkit_desc=rdkit_desc,
                     extra_desc=extra_desc)
            self.feature_field = [feature_field] + extra_desc

        elif featurizer is not None:
            self.featurizer = eval(featurizer)
        else:
            self.featurizer = dc.feat.ConvMolFeaturizer(use_chirality=False)

    def __call__(self,
                 dataset_file=None,
                 tasks=None,
                 feature_field=None,
                 id_field=None,
                ):

        # If these are not given, set them to default values for class instance:
        if dataset_file is None:
            dataset_file = self.dataset_file
        if tasks is None:
            tasks = self.tasks
        if feature_field is None:
            feature_field = self.feature_field
        if id_field is None:
            id_field = self.id_field

        loader = dc.data.CSVLoader(tasks=tasks, 
                                   feature_field=feature_field, 
                                   id_field=id_field, 
                                   featurizer=self.featurizer)

        dataset = loader.create_dataset(dataset_file)

        # If no y data given, have to modify the metadata to change the 
        # value for 'y' and 'w' from np.nan to None as this is the value 
        # checked by the get_shard() function (should probably change 
        # the get_shard function in deepchem source code really):
        if len(tasks) == 0:
            dataset.metadata_df.loc[:, 'y'] = None
            dataset.metadata_df.loc[:, 'w'] = None

        # Get number of atom features:
        def get_n_feat(data_block):
            if type(data_block) == dc.feat.mol_graphs.ConvMol:
                n_feat = data_block.n_feat
            elif type(data_block) == dc.feat.mol_graphs.WeaveMol:
                n_feat = data_block.n_features
            else:
                n_feat = None
            return n_feat

        if len(dataset.X.shape) == 1:
            n_feat = get_n_feat(dataset.X[0])
        else:
            n_feat = []
            for data_block in dataset.X[0]:
                n_feat.append(get_n_feat(data_block))

        self.additional_params['n_atom_feat'] = n_feat

        if type(dataset.X[0]) == dc.feat.mol_graphs.WeaveMol:
            self.additional_params['n_pair_feat'] = dataset.X[0].get_pair_features().shape[1]

        if self.additional_params['mode'] == 'classification':
            self.additional_params['n_classes'] = len(np.unique(dataset.y))

        return dataset, self.additional_params


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


#class mol_desc_scaling_selection(dc.trans.Transformer):
#    """
#    """
#
#    # Returns function for subselection of features:
#    class subselect():
#        def __init__(self, n_feats):
#            self.n_feats = n_feats
#        def __call__(self, x):
#            return x[:,:self.n_feats]
#
#    def __init__(self,
#                 transform_X=True,
#                 transform_y=False,
#                 transform_w=False,
#                 transform_ids=False,
#                 dataset=None,
#                 scaler='Standard',
#                 selection_method=None,
#                 n_components=-1,
#                 explained_var=1,
#                 # Update results with number of values kept:
#                 training_info={}):
#
#        super(mol_desc_scaling_selection, self).__init__(transform_X=transform_X,
#                                                         transform_y=transform_y,
#                                                         transform_w=transform_w,
#                                                         transform_ids=transform_ids,
#                                                         dataset=dataset)
#
#        self.func_ls = []
#
#        # Read descriptor values:
#        train_desc = pd.DataFrame(data=np.stack(dataset.X[:,1]),
#                                  index=dataset.ids)
#
#        if selection_method == 'PCA':
#            # Scale before PCA:
#            pre_scaler = StandardScaler()
#            train_desc = pd.DataFrame(data=pre_scaler.fit_transform(train_desc),
#                                      index=train_desc.index)
#            self.func_ls.append(pre_scaler.transform)
#
#            # Do PCA transformation:
#            train_desc_pca = PCA()
#            train_desc = pd.DataFrame(data=train_desc_pca.fit_transform(train_desc),
#                                      index=train_desc.index)
#            self.func_ls.append(train_desc_pca.transform)
#
#            # Choose number of components to keep:
#            n_feats = None
#            if n_components > 0:
#                n_feats = n_components
#            elif 0 < explained_var < 1:
#                n_feats = np.argwhere(np.cumsum(train_desc_pca.explained_variance_ratio_) > explained_var)
#                # Take index of first value which goes above cutoff and convert from array to int:
#                n_feats = int(n_feats[0].squeeze())
#
#            # Save number of features:
#            training_info['N_PCA_feats'] = n_feats
#            train_desc = train_desc.iloc[:,:n_feats]
#
#            # Works with lambda, but cannot be pickled:
#            #self.subselect = lambda x: x[:,:n_feats]
#            #self.subselect = subselect(n_feats)
#            self.func_ls.append(self.subselect(n_feats))
#
#        elif selection_method == 'PLS':
#            raise NotImplemented
#
#        # Scaling:
#        if scaler == 'Standard' or scaler == 'Standard_tanh':
#            post_scaler = StandardScaler()
#        elif scaler == 'MinMax':
#            post_scaler = MinMaxScaler(feature_range=(-1, 1))
#        train_desc = pd.DataFrame(data=post_scaler.fit_transform(train_desc),
#                                  index=train_desc.index)
#        self.func_ls.append(post_scaler.transform)
#        if scaler == 'Standard_tanh':
#            #train_desc = train_desc.applymap(lambda i: math.tanh(i))
#            self.func_ls.append(np.tanh)
#
#    def transform_array(self, X, y, w, ids):
#        X_mol = np.stack(X[:,1])
#        for f in self.func_ls:
#            X_mol = f(X_mol)
#        X = np.array(list(zip(X[:,0], list(X_mol))))
#        return (X, y, w, ids)
#
#    
#    if 'N_PCA_feats' in run_results['training_info'].index:
#        run_results[('training_info', 'N_PCA_feats')] = additional_params.get('N_PCA_feats')


def reshard(dataset, reshard_size=False):
    """
    Reshard dataset if reshard_size is given.
    """
    if reshard_size:
        dataset.reshard(reshard_size)


def do_transforms(train_set,
                  val_set=None,
                  test_set=None,
                  ext_test_set={}, 
                  transformers=[],
                  reshard_size=False,
                  parallel=False):
    """
    Do tranformations on datasets.
    """

    # Make transformations:
    for transformer in transformers:
        train_set = transformer.transform(train_set, parallel=parallel)
        if val_set:
            val_set = transformer.transform(val_set, parallel=parallel)
        if test_set:
            test_set = transformer.transform(test_set, parallel=parallel)
        for set_name in ext_test_set.keys():
            ext_test_set[set_name] = transformer.transform(ext_test_set[set_name], parallel=parallel)

    # Reshard if reshard_size is given:
    reshard(train_set, reshard_size)
    if val_set:
        reshard(val_set, reshard_size)
    if test_set:
        reshard(test_set, reshard_size)
    for set_name in ext_test_set.keys():
        reshard(ext_test_set[set_name], reshard_size)


def transform(train_set,
              val_set=None,
              test_set=None,
              ext_test_set={},
              mode='regression',
              normalise_y=True,
              reshard_size=None,
              **kwargs):
    """
    Function to transform input data before training.
    """

    transformers = []

    # Normalise transformer:
    if (mode == 'regression') and normalise_y:
        transformer = dc.trans.NormalizationTransformer(transform_y=True,
                                                        dataset=train_set)
        transformers.append(transformer)

    ## Scale any molecular descriptors and optionally 
    ## use PCA/PLS to select descriptors:
    #if 'feature_selection' in run_input.keys():
    #    feat_transformer = \
    #    mol_desc_scaling_selection(dataset=train_set,
    #                               **run_input['feature_selection'],
    #                               training_info=additional_params)
    #    transformers.append(feat_transformer)

    do_transforms(train_set=train_set,
                  val_set=val_set,
                  test_set=test_set,
                  ext_test_set=ext_test_set,
                  transformers=transformers,
                  reshard_size=reshard_size,
                  # Cannot parallelise tranformers if training separate
                  # models is already parallelised:
                  parallel=False)

    return transformers
