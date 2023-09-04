"""
Script to package final DeepChem model.
"""

import sys, os
import numpy as np
import pandas as pd
import pickle as pk
import tarfile
import glob
from datetime import datetime

# Import these modules to save version numbers
# to final code:
import sklearn
from openbabel import openbabel
import openeye as oe

from cryptography.fernet import Fernet
from rdkit import Chem
import rdkit
from sklearn.metrics import mean_squared_error
import deepchem as dc

import json

# Update this when code is moved to programs folder:
#sys.path.insert(0, '/users/xpb20111/programs/deepchem_dev_nested_CV')
#from CSVLoaderPreprocess import CSVLoaderPreprocess
#from modified_deepchem.PreprocessFeaturizerWrapper import PreprocessFeaturizerWrapper
#from predictive_models.PreprocessFeaturizerWrapper import PreprocessFeaturizerWrapper

sys.path.insert(0, '/users/xpb20111/programs/ml_model_code/2023.7.1/predictive_models/')
from ml_model_gcnn import get_chksum
from PreprocessFeaturizerWrapper import PreprocessFeaturizerWrapper

sys.path.insert(0, '/users/xpb20111/programs/deepchem_dev_nested_CV/')
from training_utils.encryption import encrypt_data

json_infile = sys.argv[1]

run_input = json.load(open(json_infile, 'r'))

# Value column should be a list to allow multitask models:
if isinstance(run_input["training_data"]["value_column"], str):
    run_input["training_data"]["value_column"] = [run_input["training_data"]["value_column"]]

# Train models separately...
# ==========================


# Collect trained models:
# =======================

# Directory to save checkpoint files to:
if not os.path.isdir(run_input["output_options"]["ckpt_file_dirname"]):
    os.mkdir(run_input["output_options"]["ckpt_file_dirname"])


# Save predictions from each individual model:
# ============================================

final_model = {}
indiv_models = []

run_dir = run_input['trained_model']['run_directory']

df_models = pd.read_csv(run_input['trained_model']['selected_models'], 
                        header=[0, 1], sep=';')

selected_models = []

for row_i, row in df_models.iterrows():

    print("Processing data for model {}".format(row_i))

    hyperparams = row['hyperparams'].astype(str).apply(eval).to_dict()

    if 'learning_rate' in hyperparams:
        learning_rate = hyperparams['learning_rate']
        hyperparams['optimizer'] = "dc.models.optimizers.Adam(learning_rate="+str(learning_rate)+")"
        del hyperparams['learning_rate']

    selected_models.append(row['model_info'][['resample_number', 'cv_fold', 'model_number']].to_list())

    model_dir = run_dir + '/' + 'resample_{}_cv_{}_model_{}'.format(
        *row['model_info'][['resample_number', 'cv_fold', 'model_number']].to_list())

    ckpt_file = glob.glob(model_dir+"/*.index")
 
    # If not found check for tar files:
    #if len(ckpt_file) == 0:
    #    if os.path.isfile(model_dir+'/trained_models_data.tar.gz'):
    #        tarfile = tarfile.open(model_dir+'/trained_models_data.tar.gz', 'r:gz')
    #        extractdir = []
    #        for tarredfile in tarfile.getmembers():
    #            if tarredfile.name.startswith('trained_model'+str(mod_i)):
    #                extractdir.append(tarredfile)
    #        if len(extractdir) == 0:
    #            sys.exit('ERROR')
    #        tarfile.extractall(path=model_dir, members=extractdir)
    #        ckpt_file = glob.glob(model_dir+"/trained_model"+str(mod_i)+"/*.index")
    #    else:
    #        sys.exit('ERROR')
    #elif len(ckpt_file) > 1:
    #    sys.exit('ERROR')
 
    # Rename checkpoint files:
    ckpt_filename = ckpt_file[0].split('/')[-1].split('.')[0]
    new_ckpt_filename = run_input["output_options"]["ckpt_filename_prefix"]+str(row_i)
    ckpt_file_dir = run_input["output_options"]["ckpt_file_dirname"]
    for file_ext in ['.index', '.data-00000-of-00001']:
        os.system('cp '+model_dir+"/"+ckpt_filename+file_ext+' '+ckpt_file_dir+'/'+new_ckpt_filename+file_ext)

    # Get checksums for checkpoint files:
    model_ckpt_file_chksums = {'.index' : get_chksum(ckpt_file_dir+'/'+new_ckpt_filename+'.index'),
                               '.data-00000-of-00001' : get_chksum(ckpt_file_dir+'/'+new_ckpt_filename+'.data-00000-of-00001')}

    # Load transformers:
    transformers = pk.load(open(model_dir+"/transformers.pk", 'rb'))

    # Collect model data:
    indiv_models.append(#{**model_data,
                        {'model_i' : row_i,
                         'mode' : row[('dataset', 'mode')],
                         'uncertainty' : bool(row[('training_info', 'uncertainty')]),
                         'transformers' : transformers,
                         'hyperparams' : hyperparams,
                         'model_fn_str' : row[('training', 'model_fn_str')],
                         'model_fn_args' : {'n_tasks' : 1, 
                                            'mode' : row[('dataset', 'mode')], 
                                            'uncertainty' : bool(row[('training_info', 'uncertainty')]), 
                                            'number_atom_features' : int(row[('training_info', 'n_atom_feat')])},
                         'model_ckpt_file' : ckpt_file_dir+'/'+new_ckpt_filename,
                         'model_ckpt_file_checksum' : model_ckpt_file_chksums})

# Read this from GCNN_info file too?
#use_chirality = False
#featurizer_fn_str = "dc.feat.ConvMolFeaturizer(use_chirality="+str(use_chirality)+")"
#featurizer = eval(featurizer_fn_str)

featurizer = PreprocessFeaturizerWrapper(smiles_column='Canon_SMILES',
                 featurizer=[dc.feat.ConvMolFeaturizer(use_chirality=True)], 
#                               dc.feat.ConvMolFeaturizer(use_chirality=True)],
                 tauto=True,
                 ph=7.4,
                 phmodel="OpenEye",
#                 rdkit_desc=True,
#                 extra_desc=[]
                 )

# Get test data:
# ==============

run_test_ids = None
run_test_preds = None
run_test_smiles = None
if run_input.get("test_data"):
    df_preds = pd.read_csv(run_input["test_data"]["predictions_filename"], 
                           index_col=[0, 1, 2, 3, 4])

    # Only take predictions from selected models:
    #df_preds = df_preds.loc[df_preds.index.get_level_values('data_split') == 'test']\
    #                   .droplevel('data_split')\
    #                   .loc[selected_models]
    df_preds = df_preds.reset_index(level=['data_split', 'task'])\
                       .loc[selected_models]

    df_preds = df_preds.loc[df_preds['task'] != 'uncertainty']

    pred_tasks = df_preds['task'].unique()

    if run_input["test_data"].get("calculate_ensemble_predictions"):
        df_av_preds = df_preds.groupby(['data_split', 'task'])\
                              .mean()\
                              .T

    df_test = pd.read_csv(run_input["test_data"]["dataset_filename"])\
                .set_index('ID', verify_integrity=True)

    # Remove uncertainty or keep uncertainty and add 0 to test set?
    #if 'uncertainty' in pred_tasks:
    #    df_test['uncertainty'] = 0

    train_set_ids = df_av_preds['train'].dropna().index
    test_set_ids = df_av_preds['test'].dropna().index

    #if isinstance(run_input["test_data"]["value_column"], str):
    #if len(pred_tasks) == 1:
    #    task = pred_tasks[0]
    #    print('Model performance: Test set RMSD:  {:.3f}\n'\
    #          '                   Train set RMSD: {:.3f}'.format(
    #          mean_squared_error(df_test.loc[test_set_ids, 
    #                                         run_input["test_data"]["value_column"]], 
    #                             df_av_preds.loc[test_set_ids, 
    #                                             'test'])**0.5,
    #          mean_squared_error(df_test.loc[train_set_ids, 
    #                                         run_input["test_data"]["value_column"]], 
    #                             df_av_preds.loc[train_set_ids, 
    #                                             'train'])**0.5))
    #else:
    if True:
        print('Model performance:')
        for task in pred_tasks: #run_input["test_data"]["value_column"]:
            print('Task:', task)
            print('\tTest set RMSD:  {:.3f}\n'\
                  '\tTrain set RMSD: {:.3f}'.format(
                  mean_squared_error(df_test.loc[test_set_ids, 
                                                 task], 
                                     df_av_preds.loc[test_set_ids, 
                                                     ('test', task)])**0.5,
                  mean_squared_error(df_test.loc[train_set_ids, 
                                                 task], 
                                     df_av_preds.loc[train_set_ids, 
                                                     ('train', task)])**0.5))

    # Only include ChEMBL data in predictions for for run_test:
    if True:
        run_test_ids = [idx for idx in test_set_ids if idx.startswith('CHEMBL')]

    run_test_preds = df_preds.drop(columns='data_split')\
                             .loc[selected_models]\
                             .groupby('task')\
                             .mean()\
                             .T\
                             .loc[run_test_ids]\
                             .to_numpy()\
                             .squeeze()

    run_test_smiles = df_test[run_input["test_data"]["smiles_column"]]\
                             .loc[run_test_ids]\
                             .to_numpy()


# Save training data:
# ===================

fps = None
datafile_chksum = None
if run_input.get("training_data"):
    extra_columns = run_input["training_data"].get('save_extra_columns')
    if extra_columns is None:
        extra_columns = []

    df_train = pd.read_csv(run_input["training_data"]["dataset_filename"])\
                 .rename(columns={
        run_input["training_data"]["smiles_column"] : 'SMILES',
        #run_input["training_data"]["value_column"] : run_input["model_info"]["prediction_value"]
        })\
                 [[run_input["training_data"]["index_column"], 
                   'SMILES'] + \
                  run_input["training_data"]["value_column"] + \
                  extra_columns]
                 #[[run_input["training_data"]["index_column"], 
                 #  'SMILES', 
                 #  run_input["model_info"]["prediction_value"]] + \
                 #  extra_columns]

    # Get fingerprints:
    df_train['RDKit_fp'] = [Chem.RDKFingerprint(Chem.MolFromSmiles(smi)) \
                            for smi in df_train['SMILES']]
    #df_train['RDKit_fp'] = \
    #df_train.apply(lambda row: Chem.RDKFingerprint(
    #                               Chem.MolFromSmiles(row['Canon_SMILES'])), 
    #                 axis=1)

    #df_train['predictions'] = 

    if run_input["training_data"].get("save_to_separate_file"):

        data_filename_ext = run_input["output_options"]\
                                     ["training_data_filename"].split('.')[-1]
        training_data_filename = run_input["output_options"]\
                                          ["training_data_filename"]
        if run_input["training_data"].get("encrypt"):
            training_data = \
            encrypt_data(df_train.to_dict(),
                         key=b'wdwykXuzZ1TLx3MmU_KYfGleJygDIF5Er6ZL-OQREeM=', 
                         key_filename=run_input["model_info"]["prediction_value"] + \
                                      '_encryption_key.txt')
            if data_filename_ext != 'pk':
                print('WARNING: Encrypted training data file must be a pk file, not {}'.format(data_filename_ext))
                training_data_filename = training_data_filename[:-len(data_filename_ext)] + 'pk'
                data_filename_ext = 'pk'
            pk.dump(training_data, open(training_data_filename, 'wb'))

        elif data_filename_ext == 'pk':
            # Save as a dictionary to avoid version issues with pandas:
            pk.dump(df_train.to_dict(), open(training_data_filename, 'wb'))

        elif data_filename_ext == 'csv':

            # Convert fingerprints to strings:
            df_train['RDKit_fp'] = [fp.ToBase64() for fp in df_train['RDKit_fp']]
            #df_train['RDKit_fp'] = [fp.ToBitString() for fp in df_train['RDKit_fp']]

            df_train.to_csv(training_data_filename, index=False)

        datafile_chksum = get_chksum(training_data_filename)
        print('Saved training data to: {}'.format(training_data_filename))

    else:
        final_model = {**final_model, 
                       'training_data' : df_train.to_dict()}


# Save model info:
# ================

final_model = {**final_model,
               **run_input['model_info'],
               'featurizer' : featurizer,
               'public_test_data' : {'ids' : run_test_ids,
                                     'smiles' : run_test_smiles,
                                     'predictions' : run_test_preds},
               'training_set_fingerprints' : fps,
               'datafile_chksum' : datafile_chksum,
               'module_versions' : {'rdkit' : rdkit.__version__,
                                    'deepchem' : dc.__version__,
                                    'sklearn' : sklearn.__version__,
                                    'openbabel' : openbabel.OBReleaseVersion(),
                                    'openeye' : oe.__version__,
                                    'python' : sys.version
                                    }
}

if len(indiv_models) > 1 or run_input["model_info"].get("ensemble"):
    final_model = {**final_model,
                   'models' : indiv_models}

else:
    del indiv_models["model_i"]
    final_model = {**final_model,
                   **indiv_models}


# Save final model in pickle file:
# ================================

# Optionally encrypt model first:
if run_input["output_options"].get("encrypt") or \
   (run_input.get("training_data") and \
    not run_input["training_data"].get("save_to_separate_file") and \
    run_input["training_data"].get("encrypt")):

    print('Encrypting model file')
    final_model = \
    encrypt_data(final_model, 
                 key=b'wdwykXuzZ1TLx3MmU_KYfGleJygDIF5Er6ZL-OQREeM=',
                 key_filename=run_input["model_info"]["prediction_value"] + \
                              '_encryption_key.txt')

pk.dump(final_model, open(run_input["output_options"]["trained_model_filename"], 'wb'))
print('Saved model to: {}'.format(run_input["output_options"]["trained_model_filename"]))
