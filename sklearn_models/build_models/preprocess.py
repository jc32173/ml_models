from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from rdkit import Chem
import pandas as pd
import numpy as np
import math
import sys

#sys.path.insert(0, '/users/xpb20111/programs/deepchem')

#from cheminfo_utils.smi_funcs import calc_rdkit_descs
#sys.path.insert(0, '/users/xpb20111/programs/cheminfo_utils')
from cheminfo_utils.smi_funcs import process_smiles #, calc_rdkit_descs
from cheminfo_utils.calc_desc import calc_desc


class GetDataset():
    """
    Class to load dataset, including preprocessing SMILES and featurizing.
    """

    def __init__(self,
                 dataset_file='', 
                 tasks=[''],
                 feature_field='',
                 id_field='',
                 descriptors=[],
                 extra_desc=[],
                 mode='regression',
                 tauto=False,
                 ph=None,
                 phmodel='OpenEye',
                 **kwargs):

        self.dataset_file = dataset_file
        if isinstance(tasks, str):
            tasks = [tasks]
        self.tasks = tasks

        if isinstance(extra_desc, str):
            extra_desc = [extra_desc]

        self.additional_params = {'n_tasks' : len(tasks), 
                                  'mode' : mode, 
                                 }

        self.feature_field=feature_field
        self.id_field=id_field
        self.descriptors = descriptors
        self.extra_desc = extra_desc
        self.tauto = tauto
        self.ph = ph
        self.phmodel = phmodel

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

        df_data = pd.read_csv(dataset_file)\
                    [[id_field, feature_field] + \
                     self.tasks + self.extra_desc]
        df_data.columns = pd.MultiIndex.from_tuples(
            [('ids', id_field), (feature_field, feature_field)] + \
            [('y', col) for col in self.tasks] + \
            [('X', col) for col in self.extra_desc])
        df_data.set_index(('ids', id_field), drop=False, inplace=True)

        # Remove any duplicate columns (e.g. if ID and feature
        # fields are both SMILES):
        df_data = df_data.loc[:,~df_data.columns.duplicated()]

        df_desc = calc_desc(df_data[feature_field])

        df_desc.columns = pd.MultiIndex.from_tuples(
            [('X', col) for col in df_desc.columns])

        df_dataset = pd.merge(left=df_data,
                              left_on=[(feature_field, feature_field)],
                              right=df_desc,
                              right_index=True,
                              how='inner')

#        if self.additional_params['mode'] == 'classification':
#            self.additional_params['n_classes'] = len(np.unique(dataset.y))

        return df_dataset, self.additional_params
