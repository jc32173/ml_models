"""
Classes and functions for Deepchem GCNN predictive models.
"""


__version__ = '2023.6.2'


from abc import ABC, ABCMeta, abstractmethod
import hashlib
import os
import sys
import pickle as pk
import numpy as np
import pandas as pd
from copy import deepcopy

from cryptography.fernet import Fernet
from rdkit import Chem
from rdkit.DataStructs import FingerprintSimilarity
# Need openbabel for some pH conversions:
#from openbabel import openbabel as ob
import deepchem as dc
from typing import Sequence
import logging


# Set log level for tensorflow:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set up logger for module:
logger = logging.getLogger(__name__)
# Set logging levels, especially when debugging:
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
#logging.getLogger().setLevel(logging.DEBUG)
#logging.getLogger("tensorflow").setLevel(logging.DEBUG)


# Add the directory containing this script to the path so that 
# other code in this directory can be imported:
#sys.path.append(os.path.dirname(__file__))
#print(sys.path)
#print(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../modified_deepchem/')))
#sys.exit()
from modified_deepchem.wrap_KerasModel_for_uncertainty import \
    wrap_KerasModel_for_uncertainty


# Decrypt encrypted pickled data:
def decrypt_data(key, encrypted_data):
    """
    Function to decrypt and unpickle data.
    """
    f = Fernet(key.encode("utf-8"))
    decrypted_data = f.decrypt(encrypted_data)
    data = pk.loads(decrypted_data)
    return data


# Taken from: 
# https://stackoverflow.com/questions/3431825/generating-an-md5-checksum-of-a-file
def get_chksum(filename):
    """
    Function to get the MD5 checksum for a file.
    """
    hash_md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    chksum = hash_md5.hexdigest()
    return chksum


# Round number of array to x dp, with 0.5 always rounded up:
def round_5up(vals, dp=1):
    """
    Function to round decimals, with 0.5 always rounded up.
    """
    # np.floor returns a float so need to convert this to an int to 
    # truncate to number of decimal places:
    if dp is not None:
        return np.floor((vals*(10**(dp))) + 0.5).astype(np.int)/(10**(dp))
    else:
        return vals


# Base class for DeepChem ML models:
class ML_Model_DC(ABC):
    """
    Class for single Deepchem GCNN model.
    """

    def __init__(self,
                 model_data,
                 encryption_key=None,
                 model_dir=''):
        """
        Load pretrained DeepChem GCNN model.
        """

        # Remove this eventually as loading from a pk file is now a classmethod:
        if isinstance(model_data, str):
            model_dir = os.path.dirname(os.path.realpath(model_data))+'/'
            model_data_file = model_data
            model_data = pk.load(open(model_data_file, 'rb'))
            # Decrypt file:
            if encryption_key is not None:
                model_data = decrypt_data(encryption_key, model_data)
            #elif isinstance(model_data, bytes):
            try:    
                model_data['model_data_file'] = model_data_file
            except TypeError: # as e:
                raise ValueError('This model is probably encrypted, need to give the encryption key.')

        self.model_dir = model_dir

        # Load information about the model:
        self._load_model_info(model_data)

        # Load transformers and featurizers if given:
        self.transformers = model_data.get('transformers')
        self.featurizer = model_data.get('featurizer')

        # Load pretrained model:
        self._load_model(model_data)

        # Make prediction to check initialisation:
        if self.featurizer is not None:
            self.predict('C')


    def _load_model_info(self, model_data):
        """
        Load information about the model.
        """
        
        # Model type and parameters:
        # Use [] for required values and .get() for optional parameters:
        self.model_data_file = model_data.get('model_data_file')
        self.version = model_data.get('version')
        self.info = model_data.get('model_info')
        self.units = model_data.get('units')
        self.python_modules = model_data.get('module_versions')

        # These have default options for backwards compatibility:
        self.mode = model_data.get('mode')
        if self.mode is None:
            self.mode = 'regression'
        self.uncertainty = model_data.get('uncertainty')
        if self.uncertainty is None:
            self.uncertainty = False

        # Check if training dataset is available:
        #self.training_data_available = False
        self.training_data_file = model_data.get('training_data_file')
        if self.training_data_file is not None:
            self.training_data_file = self.model_dir + self.training_data_file
            if not os.path.isfile(self.model_dir + self.training_data_file):
                logger.warning('Cannot find training dataset: {}'.format(
                               self.training_data_file))


    def _load_model(self, model_data):
        """
        Load the pretrained model.
        """
        
        self.model_fn_str = model_data['model_fn_str']
        if self.model_fn_str == 'dc.models.GraphConvModel':
            model_cls = dc.models.GraphConvModel
        else:
            raise NotImplementedError(
                "Currently only accepts dc.models.GraphConvModel models")

        # Apply some wrappers to fix issues with deepchem models:

        # Wrapper to fix uncertainty prediction:
        if self.uncertainty:
            model_cls = wrap_KerasModel_for_uncertainty(model_cls)

        # Wrapper class to remove decorator in original version as this
        # causes some predictions to be much slower:
        class production_model_wrapper(model_cls):
            # Record the parent class for reference:
            self.parent_class = self.model_fn_str
            # Override _compute_model with new version 
            # without @tf.function decorator:
            # @tf.function(experimental_relax_shapes=True)
            def _compute_model(self, inputs: Sequence):
                """Evaluate the model for a set of inputs."""
                return self.model(inputs, training=False)

        model_cls = production_model_wrapper

        # Process specific hyperparameters:

        # Optimizer and learning rate:
        #optimizer = model_data['hyperparams'].get('optimizer')
        #optimizer = 'dc.models.optimizers.Adam'
        #if (optimizer is not None) and \
        #   optimizer != 'dc.models.optimizers.Adam':
        #        raise NotImplementedError(
        #            "Currently only accepts dc.models.optimizers.Adam() "+\
        #            "as an optimizer")
        learning_rate = model_data['hyperparams'].get('learning_rate')
        if learning_rate is not None:
            del model_data['hyperparams']['learning_rate']
            optimizer = dc.models.optimizers.Adam(learning_rate=learning_rate)
        else:
            optimizer = dc.models.optimizers.Adam()
        model_data['hyperparams']['optimizer'] = optimizer

        # Initialise specific model:
        self.model = model_cls(**model_data['model_fn_args'],
                               **model_data['hyperparams'])

        # The trained model is saved in deepchem checkpoint files.  Check the
        # checksums of these files (stored in the pk file) to ensure that the 
        # correct ckpt file version is being loaded:
        self.model_ckpt_file = model_data['model_ckpt_file']
        for ckpt_file_ext in ['.index', '.data-00000-of-00001']:
            curr_chksum = get_chksum(self.model_dir + self.model_ckpt_file + ckpt_file_ext)
            if curr_chksum != model_data['model_ckpt_file_checksum'][ckpt_file_ext]:
                sys.exit('ERROR: Wrong checkpoint file: {}'.format(
                    self.model_ckpt_file + ckpt_file_ext))

        # Restore weights and biases from checkpoint file:
        self.model.restore(checkpoint=self.model_dir+self.model_ckpt_file)


    @classmethod
    def load_from_pk(cls, 
                     pk_file,
                     encryption_key=None):
        """
        Load a model using data in a pickle file.
        """

        model_dir = os.path.dirname(os.path.realpath(pk_file))+'/'
        model_data = pk.load(open(pk_file, 'rb'))

        # Decrypt file:
        if encryption_key is not None:
            model_data = decrypt_data(encryption_key, model_data)

        model_data['model_data_file'] = pk_file
        
        return cls(model_data, model_dir=model_dir)


    @classmethod
    def load_from_json(cls, model_data):
        """
        Load a model using data in a json file.
        """
        
        raise NotImplementedError(
            "This will be implemented in a future verion")
        
        #if isinstance(model_info, pd.DataFrame):
        #    model_info = model_info.squeeze()
        #    if len(model_info) > 1:
        #        sys.exit('ERROR: Trying to load multiple models into a single model')
        #if isinstance(model_info, pd.Series):
        #    model_info = model_info.to_dict()
        #model_data['model_data_file'] = pk_file
        #return cls(df.to_dict())


    @classmethod
    def load_from_mem(cls, model_data, model_dir=''):
        """
        Load a model using data stored in memory in a 
        pandas DataFrame or Series or a dictionary.
        """
        
        # Need to convert data to a dictionary, 
        # (via a Series if originally a DataFrame):
        if isinstance(model_data, pd.DataFrame):
            model_data = model_data.squeeze()
            if not isinstance(model_data, pd.Series):
                raise ValueError('DataFrame should only contain 1 row.')
        if isinstance(model_data, pd.Series):
            model_data = model_data.to_dict()
        if not isinstance(model_data, dict):
            raise ValueError('Input cannot be converted to dict.')
        return cls(df.to_dict(), model_dir=model_dir)


    def featurize(self, smi):
        """
        Featurize SMILES to create input object for model.
        """
        
        warnings = ''

        feats = self.featurizer.featurize(smi)

        # Need to make feats 2D before converting it to a dataset,
        # otherwise it is not read properly:
        #if len(feats.shape) == 1:
        #    feats = [feats]
        # Need to deal with possible 0D arrays:
        if len(feats.shape) == 0:
            feats = [feats.tolist()]
        if isinstance(smi, str):
            smi = [smi]
        feats = dc.data.NumpyDataset(X=feats,
                                     ids=smi)
        return feats, warnings


    def transform(self, dataset, transformers=None):
        """
        Apply transformers to input data.
        """

        warnings = ''
        if transformers is None:
            transformers = self.transformers
        if transformers is not None:
            for trans in transformers:
                dataset = trans.transform(dataset)
        return dataset, warnings


    def make_dataset(self,
                     smi,
                     y=None,
                     ids=None,
                     w=None,
                     transform=True,
                     transformers=[]):
        """
        Generate DeepChem dataset object from SMILES and 
        optionally y, ids and w data.
        """

        warnings = ''

        feats = self.featurizer.featurize(smi)

        # Need to make feats 2D before converting it to a dataset,
        # otherwise it is not read properly:
        #if len(feats.shape) == 1:
        #    feats = [feats]
        # Need to deal with possible 0D arrays:
        if len(feats.shape) == 0:
            feats = [feats.tolist()]
        if isinstance(smi, str):
            smi = [smi]
        if ids is None:
            ids = smi
        feats = dc.data.NumpyDataset(X=feats,
                                     y=y,
                                     w=w,
                                     ids=ids)

        if transform:
            if len(transformers) > 0:
                for trans in transformers:
                    feats = trans.transform(feats)
            else:
                feats, warn = self.transform(feats)
                warnings += warn

        return feats, warnings


    def predict(self, smi, dp=1):
        """
        Predict property from SMILES or list of SMILES.
        """

        warnings = ''

        feats, warn = self.featurize(smi)
        warnings += warn

        feats, warn = self.transform(feats)
        warnings += warn

        # Add squeeze() as this should be the prediction for a single 
        # molecule, not a list (output should be 1D (tasks))
        pred_val = self.model.predict(feats, transformers=self.transformers).squeeze() #.item()

        if np.ndim(pred_val) == 0:
            pred_val = np.float(pred_val)

        # Round to dp decimal places (with 0.5 always rounded up):
        if dp is not None:
            pred_val = round_5up(pred_val, dp)

        return pred_val, warnings


    # Get DataFrame containing training data:
    def load_training_data(self, 
                           training_data_file=None, 
                           encryption_key=None):
        """
        Load training dataset from file.
        """

        if training_data_file is None:
            training_data_file = self.training_data_file
        if training_data_file is None:
            raise ValueError("Cannot find training dataset.")
        elif not os.path.isfile(training_data_file):
            raise ValueError("Cannot find training dataset: {}.".format(
                             training_data_file))
        file_ext = training_data_file.split('.')[-1]
        # Load csv file:
        if file_ext == 'csv':
            pd.read_csv(training_data_file)
        # Load pickle file:
        elif file_ext == 'pk':
            df_train = pk.load(open(training_data_file, 'rb'))
            # Decrypt file:
            if encryption_key is not None:
                df_train = decrypt_data(encryption_key, df_train)
            if isinstance(df_train, dict):
                df_train = pd.DataFrame(data=df_train)

        return df_train


    @staticmethod
    def _calc_neural_fp_similarity(fp1, fp2, metric='cosine'):
        """
        Calculate similarity between neural fingerprints
        since these are not binary.
        """

        # See metrics in: Bajusz et al., J. Cheminformatics, 2015.
        if metric == 'cosine':
            # Cosine similarity:
            sim = (np.dot(fp1, fp2))/((np.dot(fp1, fp1)*np.dot(fp2, fp2))**0.5)

        elif metric == 'tanimoto':
            # Continuous Tanimoto similarity:
            sim = (np.dot(fp1, fp2))/(np.dot(fp1, fp1) + np.dot(fp2, fp2) - np.dot(fp1, fp2))

        return sim


    # Applicability domain based on chemical fingerprints:
    def applicability_domain(self, 
                             smi, 
                             nn=3, 
                             fp_type='rdkit_fp', 
                             metric='', 
                             training_data_file=None):
        """
        Calculate average similarity to the training set.
        """

        df_train = self.load_training_data(training_data_file)

        if isinstance(smi, str):
            smi = [smi]

        # Calculate fingerprint and similarities to all 
        # fingerprints in the training set:

        if fp_type == 'rdkit_fp':
            input_fps = [Chem.RDKFingerprint(Chem.MolFromSmiles(s))
                         for s in smi]

            # Get fingerprints for the training set:
            if 'rdkit_fp' in df_train.columns:
                train_fps = df_train['rdkit_fp']
            else:
                # Calculate RDKit fps:
                train_fps = [Chem.RDKFingerprint(Chem.MolFromSmiles(smi)) 
                             for smi in df_train['SMILES']]
            
            # Tanimoto similarity:
            sims = np.array([[FingerprintSimilarity(fps_i, fps_t)
                              for fps_t in train_fps] 
                             for fps_i in input_fps])

        elif fp_type == 'neural_fp':
            input_fps = self.predict_embedding(smi)

            # Set default metric:
            if metric == '':
                metric = 'cosine'

            if 'neural_fp' in df_train.columns:
                train_fps = df_train['neural_fp']
            else:
                # Calculate neural fps:
                train_fps = self.predict_embedding(df_train['SMILES'])

            sims = np.array([[self._calc_neural_fp_similarity(fps_i, fps_t, 
                                                              metric=metric)
                              for fps_t in train_fps]
                             for fps_i in input_fps])
        else:
            raise NotImplementedError(
                'Similarity to training set using {} fingerprints'+\
                'not currently implemented'.format(fp_type))

        # Calculate average similarity to most similar neighbours:
        av_sim_nn = np.mean(np.sort(sims, axis=1)[:,-nn:], axis=1)
        return av_sim_nn


    def run_test(self, encryption_key=None):
        """
        Make predictions on limited test set to check that and 
        compare to predictions made on the same molecules during
        training to ensure predictions are consistent and model
        is performing as expected.
        """

        model_data = pk.load(open(self.model_data_file, 'rb'))

        # Decrypt file:
        if encryption_key is not None:
            model_data = decrypt_data(encryption_key, model_data)

        # Original predictions based on training data calculated when 
        # model was trained:
        test_data = model_data.get('public_test_data')
        if test_data is None:
            raise ValueError('No test data available for this model.')

        smis = test_data['smiles']
        ori_preds = test_data['predictions']

        # Make new predictions from training data:
        new_preds = self.predict(smis, dp=None)[0]

        # Compare original and new predictions:
        preds_diff = new_preds - ori_preds
        print('Test results:')
        print('Difference between original and current predictions:')
        print('RMSD: {}'.format((np.mean(preds_diff**2))**0.5))
        print('Average: {}'.format(np.mean(preds_diff)))
        max_i = np.argmax(np.abs(preds_diff))
        print('Max difference: {} (Index: {})'
            .format(preds_diff[max_i], max_i))


    def predict_uncertainty(self, smi, dp=None):
        """
        Predict uncertainty based on:
        NOTE: the predicted value is not the same as from .predict(), 
        since it is calculated with dropouts turned on.
        """

        if not self.uncertainty:
            raise ValueError('This model cannot calculate uncertainty.')

        feats, warnings = self.featurize(smi)

        feats, warnings = self.transform(feats)

        # Add squeeze() as this should be the prediction for a single 
        # molecule, not a list (output should be 1D (tasks))
        av_pred_val, uncert, uncert_epistemic, uncert_aleatoric = \
            self.model.predict_uncertainty(feats)

        av_pred_val = dc.trans.undo_transforms(av_pred_val, self.transformers)

        if np.ndim(av_pred_val) == 0:
            av_pred_val = np.float(av_pred_val)

        # Round to dp decimal places (with 0.5 always rounded up):
        if dp is not None:
            av_pred_val = round_5up(av_pred_val, dp)

        #return av_pred_val, uncert, warnings
        return av_pred_val, uncert, uncert_epistemic, uncert_aleatoric, warnings


    def predict_aleatoric_uncertainty(self, smi):
        """
        Predict aleotoric uncertainty without applying Monte Carlo
        dropouts, this is ~50x faster, but not identical to 
        predict_uncertainty() since value is not averaged over
        dropouts.
        """

        if not self.uncertainty:
            raise ValueError('This model cannot calculate uncertainty.')

        feats, warnings = self.featurize(smi)
        dataset, warnings = self.transform(feats)

        # Based on predict_uncertainty() function, but without
        # loop over dropout masks:
        generator = self.model.default_generator(
            dataset, mode='uncertainty', pad_batches=False)
                                                   # Uncertainty
        # Set uncertainty = False to ensure dropout layers are
        # switched off:
        pred, var = self.model._predict(generator, [], None, False, 
                                        ['prediction', 'variance'])

        uncert_aleatoric = np.sqrt(var)

        #uncert_aleatoric = dc.trans.undo_transforms(var, self.transformers)

        #lower_bound = dc.trans.undo_transforms(pred - uncert_aleatoric, self.transformers)
        #upper_bound = dc.trans.undo_transforms(pred + uncert_aleatoric, self.transformers)

        pred = dc.trans.undo_transforms(pred, self.transformers)

        return pred, uncert_aleatoric #, lower_bound, upper_bound


    def predict_embedding(self, smi):
        """
        Predict embedding to return the neural fingerprint.
        """

        if isinstance(smi, str):
            smi = [smi]
        feats, warnings = self.featurize(smi)
        feats, warnings = self.transform(feats)
        # Length of predict_embedding is a multiple of batch_size, 
        # so need to remove the padding:
        neural_fps = self.model.predict_embedding(feats)[:len(smi)]
        return neural_fps


class Ensemble_Model_DC(ML_Model_DC):
    """
    Class for ensemble of ML_Model_DC models.
    """


    def _load_model(self, model_data):
        """
        Load the individual pretrained DeepChem GCNN 
        models in the overall ensemble model.
        """

        # Load individual models in ensemble:
        self.models = []
        self.n_models = len(model_data['models'])
        for model_i in range(self.n_models):
            self.models.append(ML_Model_DC(model_data['models'][model_i], 
                                           model_dir=self.model_dir))

            # If separate models don't have individual featurizers, 
            # set these to the overall featurizer:
            if not hasattr(self.models[-1], 'featurizer') or \
               self.models[-1].featurizer is None:
                self.models[-1].featurizer = self.featurizer

            self.models[-1].predict('C')


    @staticmethod
    def _calc_neural_fp_similarity(fp1, fp2, metric=''):
        raise ValueError('Neural fingerprint distance not implemented for '
                         'ensemble model.')


    def predict(self, smi, dp=None, ensemble_uncertainty=False):
        """
        Predict property from SMILES or list of SMILES.
        """

        warnings = ''
        preds = []

        # If the ensemble model has a featurizer, use this once rather once
        # rather than separately for each model to speed up predictions:
        if self.featurizer is not None:
            feats, warn = self.featurize(smi)
            warnings += warn
            for model_i in range(self.n_models):
                feats_trans, warn = \
                    self.models[model_i].transform(
                        feats, transformers=self.models[model_i].transformers)
                pred = \
                    self.models[model_i].model.predict(
                        feats_trans, 
                        transformers=self.models[model_i].transformers).squeeze()
                preds.append(pred)
                warnings += warn

        else:
            for model_i in range(self.n_models):
                pred, warn = self.models[model_i].predict(smi, dp=None)
                preds.append(pred)
                warnings += warn
        preds = np.array(preds)

        if self.mode == 'regression':
            ens_preds = np.mean(preds, axis=0)

            # Round to dp decimal places (with 0.5 always rounded up):
            if dp is not None:
                ens_preds = round_5up(ens_preds, dp)

            if ensemble_uncertainty:
                ens_stddev = np.std(preds, axis=0)
                return ens_preds, ens_stddev, warnings
            else:
                return ens_preds, warnings
        else:
            #ens_preds = np.mode(preds, axis=0)
            ## Possible options:
            #if ensemble_uncertainty:
            #    # Two possible options:
            #    # Number of votes for majority class/all votes:
            #    ens_concord = np.sum(preds == ens_preds)/preds.shape[1]
            #    # Number of votes for majority class/number of votes for next highest class:
            #    ens_concord = np.sum(preds == ens_preds)/preds.shape[1]

            #    return ens_preds, ens_concord, warnings
            #else:
            #    return ens_preds, warnings
            raise NotImplementedError(
                'predict() not currently implemented for ensemble classification model')


    def predict_uncertainty(self, smi, dp=None):
        """
        Calculate uncertainty.
        """

        if not self.uncertainty:
            raise ValueError('This model cannot calculate uncertainty.')

        warnings = ''
        av_preds = []
        uncerts = []
        uncerts_epistemic = []
        uncerts_aleatoric = []
        for model_i in range(self.n_models):
            av_pred, uncert, uncert_epistemic, uncert_aleatoric, warn = \
                self.models[model_i].predict_uncertainty(smi, dp=None)
            av_pred = av_pred.squeeze()
            uncert = uncert.squeeze()
            uncert_epistemic = uncert_epistemic.squeeze()
            uncert_aleatoric = uncert_aleatoric.squeeze()
            av_preds.append(av_pred)
            uncerts.append(uncert)
            uncerts_epistemic.append(uncert_epistemic)
            uncerts_aleatoric.append(uncert_aleatoric)
            warnings += warn
        av_preds = np.array(av_preds)
        uncerts = np.array(uncerts)
        uncerts_epistemic = np.array(uncerts_epistemic)
        uncerts_aleatoric = np.array(uncerts_aleatoric)

        if self.mode == 'regression':
            ens_preds = np.mean(av_preds, axis=0)
            ens_uncerts_std = np.std(uncerts, axis=0)
            ens_preds_std = np.std(av_preds, axis=0)

            # Combine different sources of uncertainty:
            ens_uncerts = np.sqrt(ens_uncerts_std**2 + ens_preds_std**2)

            # Round to dp decimal places (with 0.5 always rounded up):
            if dp is not None:
                ens_preds = round_5up(ens_preds, dp)
                ens_uncerts = round_5up(ens_uncerts, dp)

            return ens_preds, ens_uncerts, warnings
        else:
            raise NotImplementedError(
                "Uncertainty prediction not implemented for classification models")
