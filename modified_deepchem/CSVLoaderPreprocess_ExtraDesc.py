# Writen June 2021

from typing import List, Optional, Tuple, Iterable

import logging
import pandas as pd
import numpy as np

from deepchem.feat import Featurizer
from deepchem.data.data_loader import CSVLoader
from rdkit import Chem

import sys
# Update this when code is moved to programs folder:
sys.path.insert(0, '/users/xpb20111/Prosperity_partnership/Benchmark_models/Models/code/')
from calc_desc import process_smiles
from smi_funcs import calc_rdkit_descs

logger = logging.getLogger(__name__)

# Function to run process_smiles on a dataframe:
#def process_smiles_ls(ls, tauto=True, ph=None, phmodel=None):
#    for smi in ls:
        
class CSVLoaderPreprocess_ExtraDesc(CSVLoader):

  def __init__(self,
               #tasks: List[str],
               #featurizer: Featurizer,
               #feature_field: Optional[str] = None,
               #id_field: Optional[str] = None,
               #smiles_field: Optional[str] = None,
               #log_every_n: int = 1000,
               # Names of fields for molecule level descriptors:               
               # Need to correct this to use Optional:
               mol_features: Optional[Iterable[str]] = None,
               #mol_features=None,
               rdkit_features=False,
               # Additional arguments:
               tauto=False,
               ph=None,
               phmodel=None, #'OpenEye',
               **kwargs):
    super().__init__(**kwargs)
    #super().__init__(tasks, featurizer,
    #           feature_field: Optional[str] = None,
    #           id_field: Optional[str] = None,
    #           smiles_field: Optional[str] = None,
    #           log_every_n: int = 1000)
    self.tauto = tauto
    self.ph = ph
    self.phmodel = phmodel
    self.mol_features = mol_features
    self.rdkit_features = rdkit_features

    # May have to drop any nan descriptors here:
#    if mol_features == 'RDKit'
        

  def _featurize_shard(self,
                       shard: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Featurizes a shard of an input dataframe.

    Parameters
    ----------
    shard: pd.DataFrame
      DataFrame that holds a shard of the input CSV file

    Returns
    -------
    features: np.ndarray
      Features computed from CSV file.
    valid_inds: np.ndarray
      Indices of rows in source CSV with valid data.
    """
    logger.info("About to featurize shard.")
    if self.featurizer is None:
      raise ValueError(
          "featurizer must be specified in constructor to featurizer data/")

    # Code added here to preprocess SMILES (e.g. standardising tautomer, ph):
    # proc_smi should be pandas Series
    logger.info("Preprocessing SMILES before featurizing"+\
                " (canonicalise tautomer: "+str(self.tauto)+\
                ", adjust to pH: "+str(self.ph)+"(Method: "+str(self.phmodel)+"))")
    #proc_smis, warnings = process_smis(shard[self.feature_field]))
    # Not currently recording warnings:
    processed_smis = shard[self.feature_field].apply(
        lambda smi: process_smiles(smi,
                                   tauto=self.tauto,
                                   ph=self.ph,
                                   phmodel=self.phmodel)[0])
    #logger.warning(warnings)

    features = [elt for elt in self.featurizer(processed_smis)]

    mol_features = []
    if self.mol_features: # is not None and len(self.mol_features) > 0:
        mol_features = mol_features + [elt for elt in shard[self.mol_features].to_numpy()]

    #if self.mol_features == 'RDKit':
    if self.rdkit_features:
        desc_ls = [x[0] for x in Chem.Descriptors._descList]
        mol_features = mol_features + [calc_rdkit_descs(smi, desc_ls)[0] for smi in processed_smis]

#    features = [np.concatenate((features[i], mol_features[i])) for i in range(len(features))]
    features = [[features[i], mol_features[i]] for i in range(len(features))]
    # THIS SHOULD BE CHANGED TO CONSIDER ATOM AND MOLECULE LEVEL FEATURES:
    valid_inds = np.array(
        [1 if np.array(elt).size > 0 else 0 for elt in features], dtype=bool)
    features = [
        elt for (is_valid, elt) in zip(valid_inds, features) if is_valid
    ]
#    mol_features = [
#        elt for (is_valid, elt) in zip(valid_inds, mol_features) if is_valid
#    ]
#    return [np.array(features), np.array(mol_features)], valid_inds
    return np.array(features), valid_inds
