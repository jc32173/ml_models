from sklearn.model_selection import StratifiedShuffleSplit
from rdkit import Chem
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
import deepchem as dc
import pandas as pd
import numpy as np
import pickle as pk
import math
import sys
from itertools import product
import random
from datetime import datetime

from build_models.dc_metrics import all_metrics, calc_stddev #, y_stddev
metrics_ls = [all_metrics[m] for m in all_metrics['_order']]
from modified_deepchem.ExtendedGraphConvModel_ExtraDesc import ExtendedGraphConvModel_ExtraDesc

def get_hyperparams_grid(hyperparams, 
                         df_prev_hp=None):
    hp_names = hyperparams['_order']
    hp_vals = [hyperparams[hp] for hp in hp_names]
    for hp_combination in product(*hp_vals):
        hp_dict = dict(zip(hp_names, hp_combination))
        df_hp = pd.DataFrame(data={hp : str(hp_dict[hp]) for hp in hp_dict},
                             columns=hp_dict.keys(), 
                             index=[0])

        if df_prev_hp is not None:
            # Check whether set of hyperparameters have already been run:
            # Need all values to be strings
            #join_ls = [('hyperparams', hp) for hp in hp_names]
            if len(pd.merge(left=df_hp, left_on=hp_names,
                            right=df_prev_hp.astype(str), right_on=hp_names,
                            how='inner')) > 0:
                print('Skipping hyperparameter combination already run')
                continue

        # Modify any parameters which need it:
        if 'learning_rate' in hp_dict:
            hp_dict['optimizer'] = dc.models.optimizers.Adam(learning_rate=hp_dict['learning_rate'])
            del hp_dict['learning_rate']
    #    if dropout_type = 'all', 'last_dense', 'last_graph', 'all_dense', 'all_graph'
        if 'dropout' in hp_dict and 'graph_conv_layers' in hp_dict:
            hp_dict['dropout'] = [0]*len(hp_dict['graph_conv_layers']) + [hp_dict['dropout']]

        yield hp_dict, df_hp


def get_hyperparams_rand(hyperparams, 
                         n_iter, 
                         df_prev_hp=None):

    def rand_combination(hyperparams):
        hp_dict = {}
        for hp in hyperparams['_order']:
             hp_dict[hp] = random.choice(hyperparams[hp])
        return hp_dict

    n = 0
    while n < n_iter:
        hp_dict = rand_combination(hyperparams) 
        df_hp = pd.DataFrame(data={hp : str(hp_dict[hp]) for hp in hp_dict},
                             columns=hp_dict.keys(), 
                             index=[0])

        if df_prev_hp is not None:
            # Check whether set of hyperparameters have already been run:
            # Need all values to be strings
            #join_ls = [('hyperparams', hp) for hp in hp_names]
            if len(pd.merge(left=df_hp, left_on=hp_names,
                            right=df_prev_hp.astype(str), right_on=hp_names,
                            how='inner')) > 0:
                print('Skipping hyperparameter combination already run')
                continue

        # Modify any parameters which need it:
        if 'learning_rate' in hp_dict:
            hp_dict['optimizer'] = dc.models.optimizers.Adam(learning_rate=hp_dict['learning_rate'])
            del hp_dict['learning_rate']
    #    if dropout_type = 'all', 'last_dense', 'last_graph', 'all_dense', 'all_graph'
        if 'dropout' in hp_dict:
            hp_dict['dropout'] = [0]*len(hp_dict['graph_conv_layers']) + [hp_dict['dropout']]

        n += 1

        yield hp_dict, df_hp


# val_callback object:
class ValCallback():

    def __init__(self, 
                 n_batches_per_epoch, 
                 train_set, 
		 val_set, 
		 test_set, 
		 ext_test_set={}, 
                 transformers=[],
                 dump_fps=False,
                 log_freq=1):

        self.n_batches_per_epoch = n_batches_per_epoch
        self.log_freq = int(log_freq)
        self.dump_fps = dump_fps

        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.ext_test_set = ext_test_set

        self.transformers = transformers

        self.train_scores = []
        self.val_scores = []
        self.test_scores = []
        self.test_preds = [0]

        self.ext_test_scores = {}
        self.ext_test_preds = {}
        for set_name in ext_test_set:
            self.ext_test_scores[set_name] = []
            self.ext_test_preds[set_name] = [0]

        self.best_mse = None

    def __call__(self, mdl, stp):

        if stp % (self.n_batches_per_epoch*self.log_freq) == 0:
            self.train_scores.append(mdl.evaluate(self.train_set, 
                                                  metrics=metrics_ls, 
                                                  transformers=self.transformers))
            self.val_scores.append(mdl.evaluate(self.val_set, 
                                                metrics=metrics_ls, 
                                                transformers=self.transformers))
            if self.test_set:
                self.test_scores.append(mdl.evaluate(self.test_set, 
                                                     metrics=metrics_ls, 
                                                     transformers=self.transformers))

            if len(self.ext_test_set) > 0:
                for set_name in self.ext_test_set.keys():
                    self.ext_test_scores[set_name].append(
                        mdl.evaluate(self.ext_test_set[set_name], 
                                     metrics=metrics_ls, 
                                     transformers=self.transformers))

            print('Epoch: {}, MSE (loss): Training: {}, Validation: {}'.format(
                stp/self.n_batches_per_epoch, 
                self.train_scores[-1]['mse'], 
                self.val_scores[-1]['mse']))

            # If best epoch, save predictions:
            if np.argmin([i['mse'] for i in self.val_scores]) == len(self.val_scores) - 1:
                self.best_mse = self.val_scores[-1]['mse']

                if self.test_set:
                    self.test_preds[0] = mdl.predict(self.test_set, 
                                                     transformers=self.transformers)

                if self.dump_fps:
                    self.train_neural_fps = mdl.predict_embedding(self.train_set)
                    self.val_neural_fps = mdl.predict_embedding(self.val_set)
                    if self.test_set:
                        self.test_neural_fps = mdl.predict_embedding(self.test_set)

                mdl.save_checkpoint(max_checkpoints_to_keep=1,
                                    model_dir=mdl.model_dir)

                if len(self.ext_test_set) > 0:
                    for set_name in self.ext_test_set.keys():
                        self.ext_test_preds[set_name][0] = mdl.predict(self.ext_test_set[set_name], 
                                                                       transformers=self.transformers)


def train_model(mod_i, 
                model_fn_str, 
                hyperparams, 
                additional_params, 
                epochs, 
                train_set, 
                val_set, 
                test_set,
                ext_test_set, 
                # Not read here, but keep to avoid errors:
                hyperparam_search='',
                separate_process='',
                early_stopping=False,
                early_stopping_interval=50,
                early_stopping_threshold=0.0001,
                transformers=[], 
                out_file=None,
                dump_fps=False,
                run_results={},
                rand_seed=None):

    # Set random seeds for numpy and tensorflow to make reproducible:
    # Doesn't need to be set here as is already set in run_deepchem_models.py
    #if rand_seed is not None:
    #    np.random.seed(rand_seed+123)
    #    tf.random.set_seed(rand_seed+456)

    # Have to initialise model inside this function so that it is cleared when process stops:

    model = eval(model_fn_str)(**additional_params,
                               **hyperparams)

    n_batches_per_epoch = math.ceil((train_set.X.shape[0]/(hyperparams['batch_size'] + 0.0)))
    val_cb = ValCallback(n_batches_per_epoch,
                         train_set,
                         val_set,
                         test_set,
                         ext_test_set=ext_test_set,
                         transformers=transformers,
                         dump_fps=dump_fps)

    # Fit model:

    # Record fitting time:
    start_t = datetime.now()

    # Include GP hyperparameter optimisation here (could rewrite function to also include early stopping)
    # GP optimisation tested in: /users/xpb20111/DeepChem/GraphConvModel/GraphConvModel_code_tests.ipynb

    # Training with early stopping:
    # Based on: https://github.com/deepchem/deepchem/issues/1533
    if early_stopping:
        current_epoch = 0
        prev_mse = np.inf
        while current_epoch < epochs:

            model.fit(train_set,
                      nb_epoch=early_stopping_interval,
                      # Save checkpoints manually for best epoch:
                      max_checkpoints_to_keep=epochs,
                      checkpoint_interval=0,
                      callbacks=[val_cb])
            current_epoch += early_stopping_interval

            mse_diff = prev_mse - val_cb.best_mse
            if mse_diff < early_stopping_threshold:
                print('Early stopping: Change in best Valdation MSE '+\
                      'after epochs: {} and {} was {:.3f} (< threshold ({}))'\
                      .format(current_epoch,
                              current_epoch+early_stopping_interval,
                              mse_diff,
                              early_stopping_threshold))
                break
            prev_mse = val_cb.best_mse

    # No early stopping:
    else:
        model.fit(train_set,
                  nb_epoch=epochs,
                  # Save checkpoints manually for best epoch:
                  max_checkpoints_to_keep=epochs,
                  checkpoint_interval=0,
                  callbacks=[val_cb])

    end_t = datetime.now()
    training_t = end_t - start_t

    # Also save transformers to model_dir:
    pk.dump(transformers, open(model.model_dir+'/transformers.pk', 'wb'))

    # Output metrics from training:
    pk.dump([val_cb.train_scores, val_cb.val_scores, val_cb.test_scores], open('GCNN_training_output_model'+str(mod_i)+'.pk', 'wb'))

    # Save model details:

    best_epoch = np.argmin([i['mse'] for i in val_cb.val_scores])

    # Save stats on dataset splits:
    for split_name, split_scores in [['train', val_cb.train_scores], 
                                     ['val', val_cb.val_scores],
                                     ['test', val_cb.test_scores]]:
        if split_scores:
            for metric in all_metrics['_order']:
                run_results[(split_name, metric)] = round(split_scores[best_epoch][metric], 3)
    for split_name, split_data in [['train', train_set], 
                                   ['val', val_set],
                                   ['test', test_set]]:
        if split_data:
            # Need to think about this again:
            run_results[(split_name, 'y_stddev')] = calc_stddev(split_data.y, transformers)
        #run_results[(split_name, 'y_stddev')] = model.evaluate(split_data,
        #                                                       metrics=y_stddev,
        #                                                       transformers=transformers)
                                                #round(math.sqrt(np.var(split_data.y)), 3)

    # Save training details:
    run_results[('training_info', 'date')] = datetime.now().strftime("%Y-%m-%d %H:%M"),
    run_results[('training_info', 'training_time')] = str(training_t),
    if early_stopping and current_epoch < epochs:
        run_results[('training_info', 'best_epoch')] = str(best_epoch)+'/'+str(current_epoch)+' early_stopping'
    else:
        run_results[('training_info', 'best_epoch')] = str(best_epoch)+'/'+str(epochs)

    # Save stats on external test sets:
    for set_name, set_scores in val_cb.ext_test_scores.items():
        for metric in all_metrics['_order']:
            run_results[(set_name, metric)] = round(set_scores[best_epoch][metric], 3)

        run_results[(set_name, 'y_stddev')] = round(math.sqrt(np.var(ext_test_set[set_name].y)), 3)

    ## Add additional params to output:
    #for add_param, add_val in additional_params.items():
    #    run_results[('training_info', add_param)] = add_val
    run_results[('training_info', 'n_atom_feat')] = additional_params.get('n_atom_feat')
    if 'N_PCA_feats' in run_results['training_info'].index:
        run_results[('training_info', 'N_PCA_feats')] = additional_params.get('N_PCA_feats')

    # Write output?
    run_results[('model_info', 'model_number')] = mod_i
    out_file.write(';'.join([str(i) for i in run_results.to_list()])+'\n')

    # Ensure results are written to file after each model:
    out_file.flush()

    # Output predictions:
    if test_set:
        pk.dump([val_cb.test_set.ids, val_cb.test_preds], open('GCNN_test_preds_model'+str(mod_i)+'.pk', 'wb'))

    # Output predictions on external test set:
    for set_name in ext_test_set:
        pk.dump([val_cb.ext_test_set[set_name].ids, val_cb.ext_test_preds[set_name]], open('GCNN_ext_test_preds_model'+str(mod_i)+'_'+set_name+'.pk', 'wb'))

    # Output neural fingerprints:
    if dump_fps:
        pk.dump([[val_cb.train_set.ids, val_cb.train_neural_fps],
                 [val_cb.val_set.ids, val_cb.val_neural_fps],
                 [val_cb.test_set.ids, val_cb.test_neural_fps]], open('GCNN_neural_fps_model'+str(mod_i)+'.pk', 'wb'))

    # Clear keras internal state, otherwise this causes the job to run out of memory after a few loops:
    # https://www.tensorflow.org/api_docs/python/tf/keras/backend/clear_session

    tf.keras.backend.clear_session()
#    tf.reset_default_graph()
    del model
    del val_cb
