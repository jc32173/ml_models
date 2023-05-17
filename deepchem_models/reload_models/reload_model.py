# Get neural fps from model:
import deepchem as dc
import pickle as pk
import tarfile
import glob
import os
import re
import shutil
import sys
sys.path.insert(0, '/users/xpb20111/programs/ml_model_code/2021.3.5/predictive_models')
from ml_model_gcnn_ens import ML_Model_DC, get_chksum

# All originally taken from: /users/xpb20111/pIC50/docking_gnina/Combine_all_GNINA_docking_data.ipynb

def get_transformers(model_no, model_dir):
    """
    Load transformers from trained GCNN model by giving model directory and number:

    get_transformers(model_dir, model_no)
    
    (e.g. get_transformers(13, 'resample_0'))
    """
    filename = 'trained_model'+str(model_no)+'/transformers.pk'
    if model_dir[-1] != '/':
        model_dir += '/'
    if os.path.isfile(model_dir+filename):
        return pk.load(open(model_dir+filename, 'rb'))
    elif os.path.isfile(model_dir+'trained_models_data.tar.gz'):
        model_tarfile = tarfile.open(model_dir+'/trained_models_data.tar.gz', 'r:gz')
        return pk.load(model_tarfile.extractfile(filename))
    else:
        print('ERROR: Cannot find file: '+model_dir+filename)


def get_ckpt_files(model_no, model_dir):
    """
    Load transformers from trained GCNN model by giving model directory and number:

    get_transformers(model_dir, model_no)
    
    (e.g. get_transformers(13, 'resample_0'))
    """
    if model_dir[-1] != '/':
        model_dir += '/'
    if os.path.isdir(model_dir+'trained_model'+str(model_no)):
        ckpt_file = glob.glob(model_dir+'trained_model'+str(model_no)+'/ckpt*.index')[0].strip('.index')
        ckpt_no = int(ckpt_file.split('-')[-1])
        print(ckpt_file)
        return ckpt_no, ckpt_file
    elif os.path.isfile(model_dir+'trained_models_data.tar.gz'):
        model_tarfile = tarfile.open(model_dir+'trained_models_data.tar.gz', 'r:gz')

        ckpt_files = []
        for ckpt_file in model_tarfile.getnames():
            if ckpt_file.startswith('trained_model'+str(model_no)+'/ckpt-') and ckpt_file[-6:] == '.index':
                ckpt_no = re.search('ckpt-([0-9]+)\.index', ckpt_file).group(1)
                # Or could use findall:
                # ckpt_no = re.findall('ckpt-([0-9]+)\.index', f)[0]
                break

        tarfile_member = model_tarfile.getmember('trained_model'+str(model_no)+'/ckpt-'+str(ckpt_no)+'.index')
        model_tarfile.extract(tarfile_member)
        tarfile_member = model_tarfile.getmember('trained_model'+str(model_no)+'/ckpt-'+str(ckpt_no)+'.data-00000-of-00001')
        model_tarfile.extract(tarfile_member)
        # Move to original model directory:
        shutil.move('trained_model'+str(model_no)+'/', model_dir)
        # Add file root:
        ckpt_file = model_dir + ckpt_file[:-6]
        return ckpt_no, ckpt_file
    else:
        print('ERROR: Cannot find ckpt file for: '+model_dir+'trained_model'+str(model_no))



def reload_model(df_info_row, 
                 model_dir='', 
                 test_smis=None, 
                 model_cls=ML_Model_DC, 
                 featurizer=None):
    """
    Reload trainined DeepChem model using data from row of GCNN_info_all_models.csv file
    """

    model_no = df_info_row[('model_info', 'model_number')]

    print('Reading model number:', model_no)
    
    # Set default mode to regression:
    mode = df_info_row.get([('dataset', 'mode')])
    if mode is None:
        mode = 'regression'
    else:
        mode = mode.squeeze()
        
    uncert = df_info_row.get([('training', 'uncertainty')])
    if uncert is None:
        uncert = False
    else:
        uncert = uncert.squeeze()
    
    transformers = get_transformers(model_no, model_dir)

    # Get featurizer from dataset if not given as an argument:
    if featurizer is None:
        # This may need generalising:
        featurizer = eval(eval(df_info_row[('preprocessing', 'featurizer')])[0])

    model_fn_str = df_info_row[('training', 'model_fn_str')]

    hyperparam_dict = df_info_row['hyperparams'].to_dict()

    hyperparam_str = ", ".join([name+"="+str(val) for name, val in hyperparam_dict.items()])

    model_fn_args_str = "n_tasks={}, mode='{}', uncertainty={}, number_atom_features={}".format(
        len(eval(df_info_row[('dataset', 'tasks')])), 
        mode, 
        uncert,
        df_info_row[('training_info', 'n_atom_feat')])

    ckpt_no, model_ckpt_file = get_ckpt_files(model_no, model_dir)

    print('Loading from checkpoint:', ckpt_no)

    model_data = {'transformers' : transformers,
                  'featurizer' : featurizer,
                  'model_fn_str' : model_fn_str,
                  'hyperparams_str' : hyperparam_str,
                  'model_fn_args_str' : model_fn_args_str,
                  'model_ckpt_file' : model_ckpt_file,
                  'model_ckpt_file_checksum' : {'index' : get_chksum(model_ckpt_file+'.index'),
                                                'data'  : get_chksum(model_ckpt_file+'.data-00000-of-00001')}
                 }

    mod = model_cls(model_data=model_data)
    mod.predict('C')

    if os.path.isfile(model_dir+'trained_models_data.tar.gz'):
#        shutil.move(model_dir+'trained_model'+str(model_no)+'/', '/users/xpb20111/recycle_bin/untarred_ckpt_files/')
        shutil.rmtree(model_dir+'trained_model'+str(model_no)+'/')
        pass

    #
    if test_smis:
        preds = mod.predict(test_smis)[0]

    return mod
