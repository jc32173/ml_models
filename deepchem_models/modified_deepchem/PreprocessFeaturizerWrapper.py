from rdkit import Chem
import numpy as np
import pandas as pd
import sys
from deepchem.feat import Featurizer
import logging

from predictive_models.smi_funcs import process_smiles, calc_rdkit_descs

# Set up logger for module:
logger = logging.getLogger(__name__)


# DeepChem-like featurizer which allows SMILES to be converted 
# to a canonical tautomer and specific pH before featurizing.
class PreprocessFeaturizerWrapper(Featurizer):
    """
    Featurize SMILES for DeepChem GCNN models including 
    converting to a canonical tautomer and protonation form 
    for a specific pH.
    """

    def __init__(self,
                 featurizer=[], 
                 tauto=None,
                 ph=False, 
                 phmodel=None,
                 rdkit_desc=False, 
                 extra_desc=[], 
                 smiles_column=None):
        """
        Load featurizer.
        """

        self.featurizer = featurizer
        self.tauto = tauto
        self.ph = ph
        self.phmodel = phmodel
        self.rdkit_desc = rdkit_desc
        self.extra_desc = extra_desc

        # To get around needing to change CSVLoader:
        self.smiles_column = smiles_column


    def featurize(self, 
                  data, 
                  **kwargs):
        """
        Preprocess SMILES then featurize using featurizers and optionally 
        also include RDKit descriptors and any additional descriptors in 
        the input data.
        """

        # Unpack input if dataframe:
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            if len(data.shape) > 1 and data.shape[1] > 1:
                smiles = data[self.smiles_column].squeeze()
            # If data is 1D DataFrame, convert to Series:
            else:
                smiles = data.squeeze()
            smiles = smiles.to_list()

        # If data is a sting or list, assume it is smiles:
        elif isinstance(data, str):
            smiles = [data]
        else:
            smiles = data

        # Preprocess smiles:
        if self.tauto or self.ph:
            processed_smiles = [process_smiles(smi,
                                               tauto=self.tauto,
                                               ph=self.ph,
                                               phmodel=self.phmodel)[0]
                                for smi in smiles]
        else:
            processed_smiles = smiles

        # Use featurizers for atom level descriptors:
        if isinstance(self.featurizer, list):
            graph_feats = []
            for featurizer in self.featurizer:
                # Featurizer has **kwargs, pass any additional data as kwargs:
                graph_feats.append(featurizer(processed_smiles, **kwargs))
        else:
            graph_feats = self.featurizer(processed_smi)

        # Get molecule level descriptors:
        if self.extra_desc or self.rdkit_desc:
            mol_feats = []
            
            # Get extra descriptors from columns in input data:
            if self.extra_desc:
                mol_feats += [elt for elt in data[self.extra_desc].to_numpy()]

            # Calculate RDKit descriptors:
            if self.rdkit_desc:
                desc_ls = [x[0] for x in Chem.Descriptors._descList]
                mol_feats += [calc_rdkit_descs(smi, desc_ls)[0] for smi in processed_smiles]

            # Construct final array of features:
            feats = [[graph_feats[i][j] for i in range(len(graph_feats))] + [mol_feats[j]] 
                         for j in range(len(processed_smiles))]

        else:
            # Construct final array of features:
            feats = [[graph_feats[i][j] for i in range(len(graph_feats))] 
                         for j in range(len(processed_smiles))]

        feats = np.array(feats).squeeze()

        return feats
