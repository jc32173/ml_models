from sklearn.metrics import mean_squared_error
from rdkit import Chem
import tensorflow as tf
import deepchem as dc
import pandas as pd
import numpy as np
import pickle as pk
import math
import sys
import os
from itertools import product
import random
from datetime import datetime

from deepchem_models.build_models.dc_metrics import all_metrics, calc_stddev
from deepchem_models.build_models.dc_preprocess import transform
#from modified_deepchem.GraphConvClassifier import GraphConvClassifier


def get_hyperparams_grid(hyperparams):
    """
    Function to generate a grid of hyperparameters for full enumeration.
    """

    def hp_iter(hyperparams=hyperparams):
        hp_names = hyperparams['_order']
        hp_vals = [hyperparams[hp] for hp in hp_names]
        for hp_combination in product(*hp_vals):
            hp_dict = dict(zip(hp_names, hp_combination))

            ## Modify any parameters which need it:
            #if 'learning_rate' in hp_dict:
            #    hp_dict['optimizer'] = dc.models.optimizers.Adam(learning_rate=hp_dict['learning_rate'])
            #    del hp_dict['learning_rate']

            yield hp_dict

    return hp_iter, len(list(hp_iter()))


def get_hyperparams_rand(hyperparams,
                         n_iter,
                         rand_seed=None):
    """
    Function to generate randomly chosen sets of hyperparameters.
    """

    #if rand_seed is not None:
    #    np.random.seed(rand_seed)
    #else:
    #    rand_seed = np.random.randint(4294967296, dtype=np.uint32)

    if rand_seed is None:
        rand_seed = np.random.randint(4294967296, dtype=np.uint32)

    def hp_iter(hyperparams=hyperparams, rand_seed=rand_seed):
        seeded_random = random.Random(rand_seed)
        n = 0
        while n < n_iter:
            hp_dict = {}
            for hp in hyperparams['_order']:
                hp_dict[hp] = seeded_random.choice(hyperparams[hp], 
                                                   replace=False)
            hp_dict = rand_combination(hyperparams)

            ## Modify any parameters which need it:
            #if 'learning_rate' in hp_dict:
            #    hp_dict['optimizer'] = dc.models.optimizers.Adam(learning_rate=hp_dict['learning_rate'])
            #    del hp_dict['learning_rate']

            n += 1
        yield hp_dict

    return hp_iter, len(list(hp_iter()))


## val_callback object:
#class ValCallback_full_predictions():
#
#    def __init__(self, 
#                 n_batches_per_epoch, 
#                 #mode='regression',
#                 train_set, 
#		 val_set, 
#		 test_set, 
#                 all_metrics=[],
#		 ext_test_set={}, 
#                 transformers=[],
#                 dump_fps=False,
#                 log_freq=1,
#                 n_classes=None,
#                 n_epochs=None):
#
#        self.n_batches_per_epoch = n_batches_per_epoch
#        self.all_metrics = all_metrics
#        self.log_freq = int(log_freq)
#        self.dump_fps = dump_fps
#        self.n_classes = n_classes
#        self.n_epochs = n_epochs
#
#        self.train_set = train_set
#        self.val_set = val_set
#        self.test_set = test_set
#        self.ext_test_set = ext_test_set
#
#        self.transformers = transformers
#
#        self.train_scores = []
#        self.val_scores = []
#        self.test_scores = []
#        self.test_preds = [0]
#
#        self.ext_test_scores = {}
#        self.ext_test_preds = {}
#        for set_name in ext_test_set:
#            self.ext_test_scores[set_name] = []
#            self.ext_test_preds[set_name] = [0]
#
#        self.best_epoch = None
#        self.best_loss = None
#
#        # Time to start training each epoch:
#        self.training_t0 = datetime.now()
#
#    def __call__(self, mdl, stp):
#
#        if stp % (self.n_batches_per_epoch*self.log_freq) == 0:
#            self.train_scores.append(mdl.evaluate(self.train_set, 
#                                                  metrics=self.all_metrics,
#                                                  transformers=self.transformers,
#                                                  n_classes=self.n_classes))
#            if self.val_set:
#                self.val_scores.append(mdl.evaluate(self.val_set,
#                                                    metrics=self.all_metrics,
#                                                    transformers=self.transformers,
#                                                    n_classes=self.n_classes))
#            if self.test_set:
#                self.test_scores.append(mdl.evaluate(self.test_set, 
#                                                     metrics=self.all_metrics,
#                                                     transformers=self.transformers,
#                                                     n_classes=self.n_classes))
#
#            if len(self.ext_test_set) > 0:
#                for set_name in self.ext_test_set.keys():
#                    self.ext_test_scores[set_name].append(
#                        mdl.evaluate(self.ext_test_set[set_name], 
#                                     metrics=self.all_metrics,
#                                     transformers=self.transformers,
#                                     n_classes=self.n_classes))
#
#            # If best epoch, save predictions:
#            epoch_note = ""
#            if (self.val_set and (np.argmin([i['loss'] for i in self.val_scores]) == len(self.val_scores) - 1)) or \
#               (stp//self.n_batches_per_epoch == self.n_epochs):
#                if (self.val_set and (np.argmin([i['loss'] for i in self.val_scores]) == len(self.val_scores) - 1)):
#                    epoch_note = " (Current best)"
#                    self.best_epoch = stp // (self.n_batches_per_epoch) #*self.log_freq)
#                    self.best_loss = self.val_scores[-1]['loss']
#                else:
#                    epoch_note = "(Final epoch)"
#
#                if self.test_set:
#                    self.test_preds[0] = mdl.predict(self.test_set, 
#                                                     transformers=self.transformers)
#
#                if self.dump_fps:
#                    self.train_neural_fps = mdl.predict_embedding(self.train_set)
#                    if self.val_set:
#                        self.val_neural_fps = mdl.predict_embedding(self.val_set)
#                    if self.test_set:
#                        self.test_neural_fps = mdl.predict_embedding(self.test_set)
#
#                mdl.save_checkpoint(max_checkpoints_to_keep=1,
#                                    model_dir=mdl.model_dir)
#
#                if len(self.ext_test_set) > 0:
#                    for set_name in self.ext_test_set.keys():
#                        self.ext_test_preds[set_name][0] = mdl.predict(self.ext_test_set[set_name], 
#                                                                       transformers=self.transformers)
#
#            self.training_t1 = datetime.now()
#            # Can only report validation loss if validation set has been given:
#            if self.val_scores:
#                epoch_val_loss = self.val_scores[-1]['loss']
#            else:
#                epoch_val_loss = np.nan
#            print('Epoch: {:d}, Loss: Training: {}, Validation: {} (Time: {}){}'.format(
#                stp//self.n_batches_per_epoch,
#                self.train_scores[-1]['loss'],
#                epoch_val_loss,
#                str((self.training_t1 - self.training_t0)).split('.')[0],
#                epoch_note, flush=True))
#
#            self.training_t0 = self.training_t1

#class EarlyStopping():

# val_callback object:
class ValCallback():

    def __init__(self,
                 n_batches_per_epoch,
                 train_set,
		 val_set,
		 test_set=None,
		 ext_test_set={},
                 all_metrics=[],
                 transformers=[],
                 dump_fps=False,
                 log_freq=1,
                 n_classes=None,
                 n_epochs=None):

        self.n_batches_per_epoch = n_batches_per_epoch
        self.all_metrics = all_metrics
        self.log_freq = int(log_freq)
        self.dump_fps = dump_fps
        self.n_classes = n_classes
        self.n_epochs = n_epochs

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

        self.best_epoch = None
        self.best_loss = None

        # Time to start training each epoch:
        self.training_t0 = datetime.now()

    def __call__(self, mdl, stp):

        if stp % (self.n_batches_per_epoch*self.log_freq) == 0:
            self.train_scores.append(mdl.evaluate(self.train_set, 
                                                  metrics=self.all_metrics, 
                                                  transformers=self.transformers,
                                                  n_classes=self.n_classes))
            if self.val_set:
                self.val_scores.append(mdl.evaluate(self.val_set,
                                                    metrics=self.all_metrics,
                                                    transformers=self.transformers,
                                                    n_classes=self.n_classes))
            if self.test_set:
                self.test_scores.append(mdl.evaluate(self.test_set, 
                                                     metrics=self.all_metrics, 
                                                     transformers=self.transformers,
                                                     n_classes=self.n_classes))

            for set_name in self.ext_test_set.keys():
                self.ext_test_scores[set_name].append(
                    mdl.evaluate(self.ext_test_set[set_name], 
                                 metrics=self.all_metrics, 
                                 transformers=self.transformers,
                                 n_classes=self.n_classes))

            # If best epoch, save predictions:
            epoch_note = ""
            if (self.val_set and (np.argmin([i['loss'] for i in self.val_scores]) == len(self.val_scores) - 1)) or \
               (stp//self.n_batches_per_epoch == self.n_epochs):
                if (self.val_set and (np.argmin([i['loss'] for i in self.val_scores]) == len(self.val_scores) - 1)):
                    epoch_note = " (Current best)"
                    self.best_epoch = stp // (self.n_batches_per_epoch) #*self.log_freq)
                    self.best_loss = self.val_scores[-1]['loss']
                else:
                    epoch_note = "(Final epoch)"

                if self.test_set:
                    self.test_preds[0] = mdl.predict(self.test_set, 
                                                     transformers=self.transformers)

                if self.dump_fps:
                    self.train_neural_fps = mdl.predict_embedding(self.train_set)
                    if self.val_set:
                        self.val_neural_fps = mdl.predict_embedding(self.val_set)
                    if self.test_set:
                        self.test_neural_fps = mdl.predict_embedding(self.test_set)

                mdl.save_checkpoint(max_checkpoints_to_keep=1,
                                    model_dir=mdl.model_dir)

                if len(self.ext_test_set) > 0:
                    for set_name in self.ext_test_set.keys():
                        self.ext_test_preds[set_name][0] = mdl.predict(self.ext_test_set[set_name], 
                                                                       transformers=self.transformers)

            self.training_t1 = datetime.now()
            # Can only report validation loss if validation set has been given:
            if self.val_scores:
                epoch_val_loss = self.val_scores[-1]['loss']
            else:
                epoch_val_loss = np.nan
            print('Epoch: {:d}, Loss: Training: {}, Validation: {} (Time: {}){}'.format(
                stp//self.n_batches_per_epoch,
                self.train_scores[-1]['loss'],
                epoch_val_loss,
                str((self.training_t1 - self.training_t0)).split('.')[0],
                epoch_note, flush=True))

            self.training_t0 = self.training_t1


def train_model(train_set,
                val_set=None,
                test_set=None,
                ext_test_set={},
                model_fn_str='dc.models.GraphConvModel',
                hyperparams={},
                additional_params={},
                epochs=100,
                early_stopping=False,
                #early_stopping_patience=50,
                #early_stopping_interval=50,
                #early_stopping_threshold=0.0001,
                callback=False,
                transformers=[],
                all_metrics=[],
                run_results={},
                rand_seed=None, 
                **kwargs):

    # Modify any hyperparameters which need it:
    if 'learning_rate' in hyperparams:
        hyperparams['optimizer'] = dc.models.optimizers.Adam(learning_rate=hyperparams['learning_rate'])
        del hyperparams['learning_rate']

    # Initialise model:
    model = eval(model_fn_str)(**additional_params,
                               **hyperparams)

    # Add model specific loss function to metrics:
    if additional_params.get('uncertainty'):
        loss = dc.metrics.Metric(metric=lambda t, p: 
                                 model._loss_fn(p, t, np.ones(len(p))).numpy(), 
                                 name='loss', mode='regression')
    elif additional_params['mode'] == 'regression':
        loss = dc.metrics.Metric(metric=mean_squared_error, 
                                 name='loss', mode='regression')
    else:
        loss = dc.metrics.Metric(metric=lambda t, p: 
                                 model._loss_fn.loss._compute_tf_loss(tf.cast(t, tf.float32), 
                                                                      tf.cast(p, tf.float32)).numpy(), 
                                 name='loss', mode='classification', n_tasks=1)
        loss.classification_handling_mode = "threshold-one-hot"
    all_metrics.append(loss)

    n_batches_per_epoch = math.ceil((train_set.X.shape[0]/(hyperparams['batch_size'] + 0.0)))

    callbacks = []
    #if callback:
    #    callbacks.append(ValCallback(n_batches_per_epoch,
    #                                 train_set,
    #                                 val_set,
    #                                 test_set,
    #                                 ext_test_set=ext_test_set,
    #                                 all_metrics=all_metrics,
    #                                 transformers=transformers,
    #                                 dump_fps=dump_fps,
    #                                 n_classes=additional_params.get('n_classes'),
    #                                 n_epochs=epochs))

    # Fit model:

    # Record fitting time:
    start_t = datetime.now()

    # Training with early stopping:
    if early_stopping == True:
        raise NotImplementedError
    #    current_epoch = 0
    #    prev_loss = np.inf
    #    next_training_interval = min([early_stopping_patience, epochs])
    #    while current_epoch < epochs:

    #        batch_loss = \
    #        model.fit(train_set,
    #                  nb_epoch=next_training_interval,
    #                  # Save checkpoints manually for best epoch:
    #                  max_checkpoints_to_keep=epochs,
    #                  checkpoint_interval=0,
    #                  callbacks=callbacks)
    #        current_epoch += next_training_interval

    #        loss_diff = prev_loss - batch_loss
    #        if current_epoch == epochs:
    #            print('Reached maximum number of epochs')
    #        elif loss_diff <= early_stopping_threshold:
    #            print('Early stopping: Change in batch loss over '+\
    #                  'the last {} epochs was {:.3f} (< threshold ({}))'\
    #                  .format(early_stopping_patience,
    #                          loss_diff,
    #                          early_stopping_threshold))
    #            break
    #        prev_loss = batch_loss
    #        # Next training interval should be patience from the current best epoch:
    #        next_training_interval = min([val_cb.best_epoch + early_stopping_patience - current_epoch, 
    #                                      epochs - current_epoch])

    # No early stopping:
    else:
        model.fit(train_set,
                  nb_epoch=epochs,
                  # Save checkpoints manually for best epoch:
                  max_checkpoints_to_keep=epochs,
                  checkpoint_interval=0,
                  callbacks=callbacks)
        current_epoch = epochs

    end_t = datetime.now()
    training_t = end_t - start_t

    # Save training details:
    run_results[('training_info', 'date')] = datetime.now().strftime("%Y-%m-%d %H:%M")
    run_results[('training_info', 'training_time')] = str(training_t)
    run_results[('training_info', 'epochs')] = str(current_epoch)

    return model


def score_model(model,
                train_set=None,
                val_set=None,
                test_set=None,
                ext_test_set={},
                transformers=[],
                all_metrics=[],
                mode='regression',
                n_classes=None,
                run_results={},
                **kwargs):
    """
    Function to score trained model.
    """

    # Save stats on dataset splits:
    for split_name, dataset in {'train' : train_set, 
                                'val' : val_set,
                                'test' : test_set, 
                                **ext_test_set}.items():
        if dataset:
            scores = model.evaluate(dataset,
                                    metrics=all_metrics,
                                    transformers=transformers,
                                    n_classes=n_classes)

            for metric in all_metrics:
                run_results[(split_name, metric.name)] = round(scores[metric.name], 3)

            if mode == 'regression':
                run_results[(split_name, 'y_stddev')] = calc_stddev(dataset.y, transformers)


def save_trained_model(model, 
                       transformers=[]):
    """
    Function to save trained model parameters and transformers.
    """

    # Save transformers to model_dir:
    if len(transformers) > 0:
        pk.dump(transformers, open(model.model_dir+'/transformers.pk', 'wb'))

    # Save model:
    model.save_checkpoint(max_checkpoints_to_keep=1,
                          model_dir=model.model_dir)


def train_score_model(train_set,
                      val_set=None,
                      test_set=None,
                      ext_test_set={},
                      all_metrics=all_metrics,
                      hyperparams={},
                      additional_params={},
                      run_input={},
                      run_results={},
                      save_predictions=True,
                      save_neural_fps=False,
                      save_model=False,
                      rand_seed=None,
                      out_file=None,
                      **kwargs):
    """
    Train a new model and save scores and predictions.
    """

    # Preprocess datasets:
    #train_set, val_set, test_set, ext_test_set, transformers = \
    transformers = \
    transform(train_set=train_set,
              val_set=val_set,
              test_set=test_set,
              ext_test_set=ext_test_set,
              run_input=run_input)

#    ## Add additional params to output:
#    #for add_param, add_val in additional_params.items():
#    #    run_results[('training_info', add_param)] = add_val
#    run_results[('training_info', 'n_atom_feat')] = additional_params.get('n_atom_feat')
#    if additional_params['mode'] == 'classification':
#        run_results[('training_info', 'n_classes')] = additional_params['n_classes']
#    if 'Weave' in model_fn_str:
#        run_results[('training_info', 'n_pair_feat')] = additional_params.get('n_pair_feat')

    all_metrics = all_metrics[run_input['dataset']['mode']]

    # Train model:
    model = train_model(train_set=train_set,
                        val_set=val_set,
                        **run_input['training'],
                        hyperparams=hyperparams,
                        additional_params=additional_params,
                        all_metrics=all_metrics)

    # Score model:
    score_model(model,
                train_set=train_set,
                val_set=val_set,
                test_set=test_set,
                ext_test_set=ext_test_set,
                transformers=transformers,
                all_metrics=all_metrics,
                run_results=run_results)

    # Save model:
    if save_model:
        save_trained_model(model, transformers)

    # Save predictions:
    if save_predictions:
        pass
        #save predictions...
        #df_model_preds.to_csv() )

    # Save neural fingerprints:
    if save_neural_fps:
        try:
            neural_fps = model.predict_embedding()
        except AttributeError as e:
            print(e('Model cannot output neural fingerprints.'))

    # Write output:
    #run_results[('model_info', 'model_number')] = mod_i
    #with open(out_filename, 'a') as out_file:
    if out_file:
        out_file.write(';'.join([str(i) for i in run_results.to_list()])+'\n')

    # Clear keras internal state, otherwise this causes the job to run out of memory after a few loops:
    # https://www.tensorflow.org/api_docs/python/tf/keras/backend/clear_session
    tf.keras.backend.clear_session()
    # tf.reset_default_graph()
    del model
    #del val_cb

    return run_results
