from rdkit import Chem

import sklearn

# Import these modules to save version numbers
# to final code:
import rdkit
from openbabel import openbabel
import openeye as oe

import sys, os
import numpy as np
import pandas as pd
import pickle as pk

from datetime import datetime

# Update this when code is moved to programs folder:
#sys.path.insert(0, '/users/xpb20111/programs/deepchem')

#from ml_model_gcnn import get_chksum

from cheminfo_utils.calc_desc import calc_desc


## Either pickle function or set up function when model is reloaded:
#def get_calc_desc_fn(tauto=True,
#                     ph=False,
#                     phmodel=None,
#                     descriptors=['RDKit'],
#                     fingerprints=[],
#                     rm_na=True,
#                     rm_const=False,
#                     output_processed_smiles=False,
#                     **kwargs):
#    """
#    """
#
#    # Return function with options filled in:
#    def calculate_descriptors(smi_ls):
#        return calc_desc(smi_ls,
#                         #id_ls=True,
#                         tauto=tauto,
#                         ph=ph,
#                         phmodel=phmodel,
#                         descriptors=descriptors,
#                         fingerprints=fingerprints,
#                         rm_na=rm_na,
#                         rm_const=rm_const,
#                         output_processed_smiles=False)
##    return lambda x: calc_desc(x, options...)
#    return calculate_descriptors


def save_model(model, 
               run_input, 
               model_filename, 
               incl_training_data=False, 
               encrypt=False):

    final_model = {'trained_model' : model,
                   'model_info' : run_input['model_info']['notes'],
                   'date_trained' : datetime.now(),
                   'units' : None,
                   'model_fn_str' : run_input['training']['model_fn_str'],
                   **run_input['preprocessing'],
                   #'calculate_descriptors' : get_calc_desc_fn(**run_input['dataset'],
                   #                                           **run_input['preprocessing']),
                   #'descs' : x.columns.to_list(),
                   #'model' : trained_model,
                   #'train_descs' : x.to_numpy(),
                   'module_versions' : {'rdkit' : rdkit.__version__,
                                        'sklearn' : sklearn.__version__,
                                        #'openbabel' : openbabel.OBReleaseVersion(),
                                        'openeye' : oe.__version__,
                                        'python' : sys.version}}
    if incl_training_data:
        fps = [Chem.RDKFingerprint(Chem.MolFromSmiles(smi)) 
               for smi in dataset.ids]
        final_model = {**final_model,
                       'train_data' : pd.read_csv(data_file, sep=';'),
                       'train_smi' : dataset.ids,
                       'train_fps' : fps,
                       'train_preds' : preds}

    # Save model:
    if encrypt == True:
        final_model = encrypt_data(final_model, key=b'wdwykXuzZ1TLx3MmU_KYfGleJygDIF5Er6ZL-OQREeM=')

    pk.dump(final_model, open(model_filename, 'wb'))
