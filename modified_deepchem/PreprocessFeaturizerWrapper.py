from rdkit import Chem
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/users/xpb20111/Prosperity_partnership/Benchmark_models/Models/code/')
from calc_desc import process_smiles
from smi_funcs import calc_rdkit_descs
from deepchem.feat import Featurizer

class PreprocessFeaturizerWrapper(Featurizer):

    def __init__(self,
                 smiles_column=None, 
                 featurizers=[], 
                 tauto=None,
                 ph=False, 
                 phmodel=None,
                 rdkit_desc=False, 
                 extra_desc=[]):
        self.featurizers = featurizers
        self.tauto = tauto
        self.ph = ph
        self.phmodel = phmodel
        self.rdkit_desc = rdkit_desc
        self.extra_desc = extra_desc

        # To get around needing to change CSVLoader
        self.smi_col = smiles_column

    def featurize(self, 
                  data, 
                  #additional_mol_data=None, 
                  **kwargs, # kwargs, e.g. additional_data
                 ):

        # Unpack input if dataframe or string:

        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            if len(data.shape) > 1 and data.shape[1] > 1:
                smi = data[self.smi_col].squeeze()
            # If data is 1D DataFrame, convert to Series:
            else:
                smi = data.squeeze()

        # If data is a sting or list, assume it is smiles:
        elif isinstance(data, str):
            data = [data]
        if isinstance(data, list):
            smi = data

        # Preprocess smiles:
        if self.tauto or self.ph:
            if isinstance(smi, list):
                processed_smi = [process_smiles(smi,
                                                tauto=self.tauto,
                                                ph=self.ph,
                                                phmodel=self.phmodel)[0]
                                 for smi in data]
            elif isinstance(smi, pd.Series) or isinstance(smi, pd.DataFrame):
                processed_smi = smi.apply(
                lambda row: process_smiles(row,
                                           tauto=self.tauto,
                                           ph=self.ph,
                                           phmodel=self.phmodel)[0])
        else:
            processed_smi = smi

        #print('Number of SMILES changed in preprocessing: {}'.format(np.where()))

        # Use featurizers for atom level descriptors:
        if isinstance(self.featurizers, list):
            graph_feats = []
            for featurizer in self.featurizers:
                # featurizer has **kwargs, pass any additional data as kwargs:
                graph_feats.append(featurizer(processed_smi, **kwargs)) #additional_data)
        else:
            graph_feats = self.featurizer(processed_smi)

        # Get molecule level descriptors:
        if self.extra_desc or self.rdkit_desc:
            mol_feats = []
            if self.extra_desc: # is not None and len(self.mol_features) > 0:
                mol_feats = mol_feats + [elt for elt in data[self.extra_desc].to_numpy()]
            if self.rdkit_desc:
                desc_ls = [x[0] for x in Chem.Descriptors._descList]
                mol_feats += [calc_rdkit_descs(smi, desc_ls)[0] for smi in processed_smi]

            # Construct final array of features:
            feats = [[graph_feats[i][j] for i in range(len(graph_feats))] + [mol_feats[j]] 
                         for j in range(len(processed_smi))]

        else:
            # Construct final array of features:
            feats = [[graph_feats[i][j] for i in range(len(graph_feats))] 
                         for j in range(len(processed_smi))]

        feats = np.array(feats).squeeze()

        return feats
