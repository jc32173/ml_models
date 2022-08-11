from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from rdkit import Chem
import deepchem as dc
import pandas as pd
import numpy as np
import math
import sys

sys.path.insert(0, '/users/xpb20111/programs/deepchem')

from modified_deepchem.CSVLoaderPreprocess import CSVLoaderPreprocess
from modified_deepchem.CSVLoaderPreprocess_ExtraDesc import CSVLoaderPreprocess_ExtraDesc
from modified_deepchem.ConvMolRISMFeaturizer import ConvMolRISMFeaturizer
from modified_deepchem.PreprocessFeaturizerWrapper import PreprocessFeaturizerWrapper

# Check for overlap between sets using:
def check_data_split(train_set, val_set, test_split):
    if (set(train_set.ids) & set(val_set.ids)) | \
       (set(train_set.ids) & set(test_set.ids)) | \
       (set(val_set.ids) & set(test_set.ids)):
        sys.exit('ERROR: Overlap between train, validation and test sets.')


def get_lipo_split(dataset):
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

    check_data_split(train_set, val_set, test_split)

    return train_set, val_set, test_split

# Get dataset using InMemoryLoader so that RF-Score descriptors can be 
# read from the properties of RDKit mol objects and included as atom level 
# features:
# Different to CSVLoader datasets, so will have to be run separately, not part of that class
#def add_rf_score_desc(dataset_file='',
#                      tasks=[''],
#                      feature_field='',
#                      id_field='',
#                      featurizer=dc.feat.ConvMolFeaturizer,
#                      mode='regression',
#                      use_chirality=False,
#                      tauto=False,
#                      ph=None,
#                      phmodel='OpenEye',
#                      extra_desc=None,
#                      DAGModel=False,
#                      encoding='one-hot',
#                      featurizer=EmptyConvMolFeaturizer):
#
#    # Get dataset using InMemoryLoader so that RF-Score descriptors can be 
#    # read from the properties of RDKit mol objects and included as atom level 
#    # features:
#    mol_ls = []
#    y_ls = []
#    w_ls = []
#    id_ls = []
#    data_ls = []
#    for line in open(dataset_file, 'r').readlines():
#        data_ls.append(line.strip().split(','))
#    for (molname, y, sdf_file, rf_desc_file) in data_ls:
#        y = float(y)
#        mol = Chem.MolFromMolFile(sdf_file)
#
#        # Give option to convert to RDKit tautomer/OpenEye pH?
#        # Maybe not really necessary since tautomer/pH state already chosen during docking
#
#        #rf_desc_file = rf_score_desc_dir+'/RF-Score_desc_atomic_'+molname+'.csv'
#        df_rf_desc = pd.read_csv(rf_desc_file)
#        df_rf_desc.set_index('atomnum', verify_integrity=True, inplace=True)
#        for atom in mol.GetAtoms():
#            ai = atom.GetIdx()
#            at = atom.GetSymbol()
#            # Check atom symbol as a check that atom indexes are the same in sdf and RF_desc files:
#            if df_rf_desc.loc[ai, 'atomname'] != at:
#                print('ERROR', at, df_rf_desc.loc[ai, 'atomname'], ai)
#                mol = None
#                break
#
#            # Different ways to include RF-Score descriptors, either as raw values, normalised or 
#            # one-hot encoded:
#            
#            # Including raw RF-Score descriptor values:
#            if encoding == 'raw':
#                for prpty in ['6', '7', '8', '16']:
#                    rfd = int(df_rf_desc.loc[ai, prpty].squeeze())
#                    #atom.SetIntProp(key='rf_desc_'+prpty, 
#                    #                val=rfd)
#                    # Has to be added as a molecule level property, not atom level:
#                    mol.SetIntProp(key='atom {:08d} rf_desc_{}'.format(ai, prpty), 
#                                   val=rfd)
#
#            # Normalise RF-Score descriptors:
#            elif encoding == 'norm':
#                for prpty, norm_fac in [('6', 240), ('7', 70), ('8', 60), ('16', 6)]:
#                    rfd = int(df_rf_desc.loc[ai, prpty].squeeze())/norm_fac
#                    # Has to be added as a molecule level property, not atom level:
#                    # If normalised has to be "Double" property:
#                    mol.SetDoubleProp(key='atom {:08d} rf_desc_{}'.format(ai, prpty),
#                                   val=rfd)
#        
#            # One-hot encode RF-Score descriptors:
#            elif encoding == 'one-hot':
#                for prpty, (bin_width, n_bins) in [('6', (40, 6)), ('7', (10, 7)), ('8', (10, 7)), ('16', (1, 6))]:
#                    onehot = np.zeros(n_bins, dtype=int)
#                    rfd = int(df_rf_desc.loc[ai, prpty].squeeze())
#                    rfd_bin = rfd//bin_width
#                    if rfd_bin > n_bins - 1:
#                        print('WARNING: Value for rf_desc_{} ({}) is greater than the bin range ({}), will add to largest bin.'.format(prpty, rfd, bin_width*n_bins))
#                        print(rfd)
#                        rfd_bin = n_bins - 1
#                    onehot[rfd_bin] = 1                        
#                    # Has to be added as a molecule level property, not atom level:
#                    for b_i, b_val in enumerate(onehot):
#                        mol.SetIntProp(key='atom {:08d} rf_desc_{}_bin{}'.format(ai, prpty, b_i),
#                                       val=int(b_val))
#                
#        # Set up lists of mol objects, y values, weights and ids for the InMemoryLoader:
#        if mol is not None:
#            mol_ls.append(mol)
#            y_ls.append(y)
#                               # n_tasks
#            w_ls.append(np.ones((1), np.float32))
#            id_ls.append(molname)
#
#    # For raw or normalised RF-Score descriptors:
#    if encoding != 'one-hot':
#        loader = dc.data.InMemoryLoader(tasks=tasks, 
#                                        featurizer=featurizer(use_chirality=use_chirality,
#                                        atom_properties=['rf_desc_6', 'rf_desc_7', 'rf_desc_8', 'rf_desc_16']))
#        dataset = loader.create_dataset(inputs=list(zip(mol_ls, y_ls, w_ls, id_ls)))
#
#    # For one-hot encoded RF-Score descriptors:
#    # Generate list of names of extra properties for one-hot encoding:
#    extra_atom_properties = []
#    for prpty, (bin_width, n_bins) in [('6', (40, 6)), ('7', (10, 7)), ('8', (10, 7)), ('16', (1, 6))]:
#        for b_i in range(n_bins):
#            extra_atom_properties.append('rf_desc_{}_bin{}'.format(prpty, b_i))
#    #print('Extra properties:')
#    #print(extra_atom_properties)
#
#    loader = dc.data.InMemoryLoader(tasks=tasks, 
#                                    featurizer=featurizer(use_chirality=use_chirality,
#                                                          atom_properties=extra_atom_properties))
#    dataset = loader.create_dataset(inputs=list(zip(mol_ls, y_ls, w_ls, id_ls)))
#
#    return dataset, x_feats


# Make this as a class:
class GetDataset():

    def __init__(self,
                 dataset_file='', 
                 tasks=[''],
                 feature_field='',
                 id_field='',
                 featurizer=None,
                 mode='regression',
                 #use_chirality=False,
                 tauto=False,
                 ph=None,
                 phmodel='OpenEye',
                 rdkit_desc=False,
                 extra_desc=[], 
                 DAGModel=False):

        self.dataset_file = dataset_file
        if isinstance(tasks, str):
            tasks = [tasks]
        self.tasks = tasks

        if isinstance(extra_desc, str):
            extra_desc = [extra_desc]

        self.additional_params = {'n_tasks' : len(tasks), 
                                  'mode' : mode}

        self.feature_field=feature_field
        self.id_field=id_field
#        self.tauto=tauto
#        self.ph=ph
#        self.phmodel=phmodel
#        self.rdkit_desc=rdkit_desc
#        self.extra_desc=extra_desc
#        self.DAGModel=DAGModel

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

        loader = self.get_data_loader(tasks, 
                                      feature_field, 
                                      id_field)

        dataset = loader.create_dataset(dataset_file)

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

        # Have to transform dataset if DAG:
        # Normalisation is done later
        #if self.DAGModel:
        #    max_atoms = max([mol.get_num_atoms() for mol in dataset.X])
        #    self.additional_params['max_atoms'] = max_atoms
        #    transformer = dc.trans.DAGTransformer(max_atoms=max_atoms)
        #    dataset = transformer.transform(dataset)
        #    # Worth resharding here? - DAG has many more graphs for each molecule?
        #    # This is done here: https://github.com/deepchem/deepchem/blob/master/examples/tox21/tox21_tensorgraph_DAG.py
        #    #dataset.reshard()
        #    transformers = [transformer]
        #else:
        #    transformers = []

        transformers = []

        return dataset, self.additional_params, transformers

    def get_data_loader(self, 
                        tasks,
                        feature_field, 
                        id_field):

        #if self.extra_desc == 'RF-Score':   

        # First version had a different CSVLoader depending on the preprocessing, 
        # have now moved preprocessing to featurizer, so use unmodified CSVLoader, 
        # but with own featurizer wrapper:
#        if self.rdkit_desc or (self.extra_desc is not None and len(self.extra_desc) > 0):
#            print('Using Preprocess_ExtraDesc')
#            loader = CSVLoaderPreprocess_ExtraDesc(tasks=tasks, 
#                                                   feature_field=feature_field, 
#                                                   id_field=id_field,
#                                                   rdkit_features=self.rdkit_desc, 
#                                                   mol_features=self.extra_desc, 
#                                                   tauto=self.tauto, 
#                                                   ph=self.ph, 
#                                                   phmodel=self.phmodel, 
#                                                   featurizer=self.featurizer)
#            if 'Weave' in str(self.featurizer):
#                get_n_feat = lambda ds: ds.X[0][0].n_features
#            else:
#                get_n_feat = lambda ds: ds.X[0][0].n_feat
#
#        elif self.ph or self.tauto:
#            print('Using CSVLoaderPreprocess')
#            loader = CSVLoaderPreprocess(tasks=tasks, 
#                                         feature_field=feature_field, 
#                                         id_field=id_field, 
#                                         tauto=self.tauto, 
#                                         ph=self.ph, 
#                                         phmodel=self.phmodel, 
#                                         featurizer=self.featurizer) 
#            if 'Weave' in str(self.featurizer):
#                get_n_feat = lambda ds: ds.X[0].n_features
#            else:
#                get_n_feat = lambda ds: ds.X[0].n_feat
#        else:
#            print('Using dc.data.CSVLoader')
#            loader = dc.data.CSVLoader(tasks=tasks, 
#                                       feature_field=feature_field, 
#                                       id_field=id_field, 
#                                       featurizer=self.featurizer) 
#            if 'Weave' in str(self.featurizer):
#                get_n_feat = lambda ds: ds.X[0].n_features
#            else:
#                get_n_feat = lambda ds: ds.X[0].n_feat
#        return loader, get_n_feat
        print('Using dc.data.CSVLoader')
        loader = dc.data.CSVLoader(tasks=tasks, 
                                   feature_field=feature_field, 
                                   id_field=id_field, 
                                   featurizer=self.featurizer)
        return loader


def train_val_test_split(dataset, 
                         split_method='rand', 
                         dataset_file=None, 
                         #id_field=None, 
                         strat_field=None, 
                         frac_train=0.7, 
                         frac_valid=0.15, 
                         frac_test=0.15, 
                         rand_seed=0, 
                         transformer='norm',
                         transformers=[],
                         reshard_size=False):

    if split_method == 'rand':

        # Random split using deepchem:

        splitter = dc.splits.RandomSplitter()
        train_set, val_set, test_set = \
            splitter.train_valid_test_split(dataset,
                                            frac_train=frac_train,
                                            frac_valid=frac_valid,
                                            frac_test=frac_test,
                                            seed=rand_seed)
    elif split_method == 'fp':

        # Split based on fingerprint similarity:

        splitter = dc.splits.FingerprintSplitter()
        train_set, val_set, test_set = \
            splitter.train_valid_test_split(dataset,
                                            frac_train=frac_train,
                                            frac_valid=frac_valid,
                                            frac_test=frac_test,
                                            seed=rand_seed)
    elif split_method == 'butina':

        # Split based on butina clustering:

        splitter = dc.splits.ButinaSplitter(cutoff=0.6)
        train_set, val_set, test_set = \
            splitter.train_valid_test_split(dataset,
                                            frac_train=frac_train,
                                            frac_valid=frac_valid,
                                            frac_test=frac_test,
                                            seed=rand_seed)
    elif split_method == 'strat':

        # Stratified split using sklearn:

        df = pd.read_csv(dataset_file)
        # If canon_SMILES is not index:
        # df.set_index(id_field, inplace=True, verify_integrity=True)
        c = df[strat_field]

        # Do two separate splits to get three sets:
        splitter = StratifiedShuffleSplit(1, 
                                          test_size=1-frac_train, 
                                          random_state=rand_seed
                                         ).split(dataset.X, c)
        train_idx, val_test_idx = list(splitter)[0]

        train_set = dataset.select(train_idx)
        val_test_set = dataset.select(val_test_idx)

        # Have to split val_test_set in second split to ensure no overlap:
        splitter = StratifiedShuffleSplit(1, 
                                          test_size=frac_test/(frac_valid+frac_test), 
                                          random_state=rand_seed+100
                                         ).split(val_test_set.X, c[val_test_idx])
        val_idx, test_idx = list(splitter)[0]

        val_set = val_test_set.select(val_idx)
        test_set = val_test_set.select(test_idx)

    elif split_method == 'predefinied_lipo':
        train_set, val_set, test_set = get_lipo_split(dataset)

    # Normalise transformer:

    if transformer == 'norm':
        transformer = dc.trans.NormalizationTransformer(transform_y=True, 
                                                        dataset=train_set)

        train_set = transformer.transform(train_set)
        val_set = transformer.transform(val_set)
        if test_set:
            test_set = transformer.transform(test_set)

        transformers.append(transformer)

    if reshard_size:
        train_set.reshard(reshard_size)
        val_set.reshard(reshard_size)
        if test_set:
            test_set.reshard(reshard_size)

    return train_set, val_set, test_set, transformers


def get_rism_dataset():
    pass

## See ExtraDesc_tests.ipynb for testing:
## Rewrite as a transformer?
#def mol_desc_scaling_selection(train_set,
#                               other_datasets=[],
#                               scaler='Standard',
#                               selection_method=None,
#                               n_components=-1,
#                               explained_var=1,
#                               # Update results with number of values kept:
#                               training_info={}):
#
#    # Read descriptor values:
#    train_desc = pd.DataFrame(data=np.stack(train_set.X[:,1]),
#                              index=train_set.ids)
#
#    if selection_method == 'PCA':
#        # Scale before PCA:
#        pre_scaler = StandardScaler()
#        train_desc = pd.DataFrame(data=pre_scaler.fit_transform(train_desc),
#                                  index=train_desc.index)
#
#        # Do PCA transformation:
#        train_desc_pca = PCA()
#        train_desc = pd.DataFrame(data=train_desc_pca.fit_transform(train_desc),
#                                  index=train_desc.index)
#
#        # Choose number of components to keep:
#        n_feats = None
#        if n_components > 0:
#            n_feats = n_components
#        elif explained_var < 1:
#            n_feats = np.argwhere(np.cumsum(train_desc_pca.explained_variance_ratio_) > explained_var)
#            # Take index of first value which goes above cutoff and convert from array to int:
#            n_feats = int(n_feats[0].squeeze())
#    
#        # Save number of features:
#        training_info['N_PCA_feats'] = n_feats
#        train_desc = train_desc.iloc[:,:n_feats]
#        
#    elif selection_method == 'PLS':
#        raise NotImplemented
#
#    # Scaling:
#    if scaler == 'Standard' or scaler == 'Standard_tanh':
#        post_scaler = StandardScaler()
#    elif scaler == 'MinMax':
#        post_scaler = MinMaxScaler(feature_range=(-1, 1))
#    train_desc = pd.DataFrame(data=post_scaler.fit_transform(train_desc),
#                              index=train_desc.index)
#    if scaler == 'Standard_tanh':
#        train_desc = train_desc.applymap(lambda i: math.tanh(i))
#
#    # Recreate the dataset:
#    def get_shards_with_new_data(dataset, new_desc):
#        for (X, y, w, ids) in dataset.itershards():
#            X = np.array(list(zip(X[:,0], list(new_desc.loc[ids].to_numpy()))))
#            yield X, y, w, ids
#
#    train_set = dc.data.DiskDataset.create_dataset(get_shards_with_new_data(train_set, train_desc), 
#                                           data_dir=train_set.data_dir, 
#                                           tasks=train_set.tasks)
#
#    # Replicate steps on other datasets (e.g. val, test):
#    other_datasets_new = []
#    for dataset in other_datasets:
#
#        # Read descriptors:
#        dataset_desc = pd.DataFrame(data=np.stack(dataset.X[:,1]),
#                                    index=dataset.ids)
#
#        if selection_method == 'PCA':
#            # Scaling before PCA:
#            dataset_desc = pd.DataFrame(data=pre_scaler.transform(dataset_desc),
#                                        index=dataset_desc.index)
#        
#            # PCA and select features:
#            dataset_desc = pd.DataFrame(data=train_desc_pca.transform(dataset_desc),
#                                        index=dataset_desc.index).iloc[:,:n_feats]
#
#        # Scaling:
#        dataset_desc = pd.DataFrame(data=post_scaler.transform(dataset_desc),
#                                    index=dataset_desc.index)
#        if scaler == 'Standard_tanh':
#            dataset_desc = dataset_desc.applymap(lambda i: math.tanh(i))
#
#        # Save new dataset:
#        dataset = dc.data.DiskDataset.create_dataset(get_shards_with_new_data(dataset, dataset_desc), 
#                                                     data_dir=dataset.data_dir, 
#                                                     tasks=dataset.tasks)
#        other_datasets_new.append(dataset)
#    return train_set, other_datasets_new, transformer



class mol_desc_scaling_selection(dc.trans.Transformer):

    # Returns function for subselection of features:
    class subselect():
        def __init__(self, n_feats):
            self.n_feats = n_feats
        def __call__(self, x):
            return x[:,:self.n_feats]

    def __init__(self,
                 transform_X=True,
                 transform_y=False,
                 transform_w=False,
                 transform_ids=False,
                 dataset=None,
                 scaler='Standard',
                 selection_method=None,
                 n_components=-1,
                 explained_var=1,
                 # Update results with number of values kept:
                 training_info={}):

        super(mol_desc_scaling_selection, self).__init__(transform_X=transform_X,
                                                         transform_y=transform_y,
                                                         transform_w=transform_w,
                                                         transform_ids=transform_ids,
                                                         dataset=dataset)

        self.func_ls = []

        # Read descriptor values:
        train_desc = pd.DataFrame(data=np.stack(dataset.X[:,1]),
                                  index=dataset.ids)

        if selection_method == 'PCA':
            # Scale before PCA:
            pre_scaler = StandardScaler()
            train_desc = pd.DataFrame(data=pre_scaler.fit_transform(train_desc),
                                      index=train_desc.index)
            self.func_ls.append(pre_scaler.transform)

            # Do PCA transformation:
            train_desc_pca = PCA()
            train_desc = pd.DataFrame(data=train_desc_pca.fit_transform(train_desc),
                                      index=train_desc.index)
            self.func_ls.append(train_desc_pca.transform)

            # Choose number of components to keep:
            n_feats = None
            if n_components > 0:
                n_feats = n_components
            elif 0 < explained_var < 1:
                n_feats = np.argwhere(np.cumsum(train_desc_pca.explained_variance_ratio_) > explained_var)
                # Take index of first value which goes above cutoff and convert from array to int:
                n_feats = int(n_feats[0].squeeze())

            # Save number of features:
            training_info['N_PCA_feats'] = n_feats
            train_desc = train_desc.iloc[:,:n_feats]

            # Works with lambda, but cannot be pickled:
            #self.subselect = lambda x: x[:,:n_feats]
            #self.subselect = subselect(n_feats)
            self.func_ls.append(self.subselect(n_feats))

        elif selection_method == 'PLS':
            raise NotImplemented

        # Scaling:
        if scaler == 'Standard' or scaler == 'Standard_tanh':
            post_scaler = StandardScaler()
        elif scaler == 'MinMax':
            post_scaler = MinMaxScaler(feature_range=(-1, 1))
        train_desc = pd.DataFrame(data=post_scaler.fit_transform(train_desc),
                                  index=train_desc.index)
        self.func_ls.append(post_scaler.transform)
        if scaler == 'Standard_tanh':
            #train_desc = train_desc.applymap(lambda i: math.tanh(i))
            self.func_ls.append(np.tanh)

    def transform_array(self, X, y, w, ids):
        X_mol = np.stack(X[:,1])
        for f in self.func_ls:
            X_mol = f(X_mol)
        X = np.array(list(zip(X[:,0], list(X_mol))))
        return (X, y, w, ids)
