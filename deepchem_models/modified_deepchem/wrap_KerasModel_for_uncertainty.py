import numpy as np
import tensorflow as tf
from deepchem.data import Dataset 
from deepchem.trans import Transformer, undo_transforms
from typing import Iterable, List, Tuple, Any, Optional
from deepchem.utils.typing import OneOrMany
import logging


# Set up logger for module:
logger = logging.getLogger(__name__)


def wrap_KerasModel_for_uncertainty(cls):
  """
  Function to generate wrapper class which can inherit from different DeepChem 
  GCNN model classes (cls).
  """

  class KerasModel_uncertainty_wrapper(cls):
    """
    Wrapper class to replace _predict() method of KerasModel to correct
    uncertainty calculation by passing training=True when the model is
    called and so ensure that dropouts are turned on.  Also includes a 
    modified version of the predict_uncertainty() function to return
    separate aleatoric and epistatic uncertainties and to ensure that
    batch normalisation layers are switched off for uncertainty prediction.
    """

    def _predict(self, generator: Iterable[Tuple[Any, Any, Any]],
                 transformers: List[Transformer],
                 outputs: Optional[Iterable[tf.Tensor]], uncertainty: bool,
                 other_output_types: Optional[Iterable[str]]):
      """
      Modified _predict() function to set training=True when predicting 
      uncertainty to turn on dropout layers and to read other_output_types
      to output specific prediction types (e.g. 'variance' for aleatoric 
      uncertainty)
      """

      logger.debug('Using corrected _predict() method')
      results: Optional[List[List[np.ndarray]]] = None
      variances: Optional[List[List[np.ndarray]]] = None
      if (outputs is not None) and (other_output_types is not None):
        raise ValueError(
            'This model cannot compute outputs and other output_types simultaneously.'
            'Please invoke one at a time.')
      if uncertainty and (other_output_types is not None):
        raise ValueError(
            'This model cannot compute uncertainties and other output types simultaneously.'
            'Please invoke one at a time.')
      if uncertainty:
        assert outputs is None
        if self._variance_outputs is None or len(self._variance_outputs) == 0:
          raise ValueError('This model cannot compute uncertainties')
        if len(self._variance_outputs) != len(self._prediction_outputs):
          raise ValueError(
              'The number of variances must exactly match the number of outputs')
      if other_output_types:
        assert outputs is None
        if self._other_outputs is None or len(self._other_outputs) == 0:
          raise ValueError(
              'This model cannot compute other outputs since no other output_types were specified.'
          )
      if (outputs is not None and self.model.inputs is not None and
          len(self.model.inputs) == 0):
        raise ValueError(
            "Cannot use 'outputs' argument with a model that does not specify its inputs."
            "Note models defined in imperative subclassing style cannot specify outputs"
        )
      if tf.is_tensor(outputs):
        outputs = [outputs]
      for batch in generator:
        inputs, labels, weights = batch
        self._create_inputs(inputs)
        inputs, _, _ = self._prepare_batch((inputs, None, None))
  
        # Invoke the model.
        if len(inputs) == 1:
          inputs = inputs[0]
        if outputs is not None:
          outputs = tuple(outputs)
          key = tuple(t.ref() for t in outputs)
          if key not in self._output_functions:
            self._output_functions[key] = tf.keras.backend.function(
                self.model.inputs, outputs)
          output_values = self._output_functions[key](inputs)
        else:
          # Modification for uncertainty prediction to
          # set training=True:
          if uncertainty:
            output_values = self.model(inputs, training=True)
          else:
            output_values = self._compute_model(inputs)
          if tf.is_tensor(output_values):
            output_values = [output_values]
          output_values = [t.numpy() for t in output_values]
  
        # Apply tranformers and record results.
        if uncertainty:
          var = [output_values[i] for i in self._variance_outputs]
          if variances is None:
            variances = [var]
          else:
            for i, t in enumerate(var):
              variances[i].append(t)
        access_values = []
        if other_output_types:
          # Modification to ensure other_output_types is read to 
          # make predictions for specific output types:
          # See: https://github.com/deepchem/deepchem/issues/3268
          for output_type in other_output_types:
            if output_type == 'prediction':
              access_values += self._prediction_outputs
            elif output_type == 'loss':
              access_values += self._loss_outputs
            elif output_type == 'variance':
              access_values += self._variance_outputs
            elif output_type == 'embedding':
              access_values += self._other_outputs
            else:
              raise ValueError('Unknown output type.')
        elif self._prediction_outputs is not None:
            access_values += self._prediction_outputs
  
        if len(access_values) > 0:
          output_values = [output_values[i] for i in access_values]
  
        if len(transformers) > 0:
          if len(output_values) > 1:
            raise ValueError(
                "predict() does not support Transformers for models with multiple outputs."
            )
          elif len(output_values) == 1:
            output_values = [undo_transforms(output_values[0], transformers)]
        if results is None:
          results = [[] for i in range(len(output_values))]
        for i, t in enumerate(output_values):
          results[i].append(t)
  
      # Concatenate arrays to create the final results.
      final_results = []
      final_variances = []
      if results is not None:
        for r in results:
          final_results.append(np.concatenate(r, axis=0))
      if uncertainty and variances is not None:
        for v in variances:
          final_variances.append(np.concatenate(v, axis=0))
        return zip(final_results, final_variances)
      if len(final_results) == 1:
        return final_results[0]
      else:
        return final_results


    # Decorator to wrapper around predict_uncertainty of KerasModel 
    # to allow batch_norm layers to be frozen.
    def freeze_batch_norm(fn):
      """
      Decorator to prevent retraining batch_norm layers when model
      is invoked with Training=True.
      """

      def freeze_batch_norm_wrapper(self, *args, **kwargs):
        logger.debug('Freezing batch_norm layers for uncertainty prediction')
        for bn in self.model.batch_norms:
          bn.trainable = False
        output = fn(self, *args, **kwargs)
        for bn in self.model.batch_norms:
          bn.trainable = True
        return output

      return freeze_batch_norm_wrapper


    @freeze_batch_norm
    def predict_uncertainty(self, dataset: Dataset, masks: int = 50
                           ) -> OneOrMany[Tuple[np.ndarray, np.ndarray]]:
      """
      Modified predict_uncertainty() function to return separate estimations
      of aleatoric and epistemic uncertainty alongside total uncertainty.
      """
      logger.debug('Using corrected predict_uncertainty()')
      sum_pred: List[np.ndarray] = []
      sum_sq_pred: List[np.ndarray] = []
      sum_var: List[np.ndarray] = []
      for i in range(masks):
        generator = self.default_generator(
            dataset, mode='uncertainty', pad_batches=False)
                                                   # Uncertainty
        results = self._predict(generator, [], None, True, None)
        if len(sum_pred) == 0:
          for p, v in results:
            sum_pred.append(p)
            sum_sq_pred.append(p * p)
            sum_var.append(v)
        else:
          for j, (p, v) in enumerate(results):
            sum_pred[j] += p
            sum_sq_pred[j] += p * p
            sum_var[j] += v
      output = []
      std = []
      # Save approximate aleatoric and epistatic losses separately:
      aleatoric = []
      epistatic = []
      for i in range(len(sum_pred)):
        p = sum_pred[i] / masks
        output.append(p)
        std.append(np.sqrt(sum_sq_pred[i] / masks - p * p + sum_var[i] / masks))
        aleatoric.append(np.sqrt(sum_var[i] / masks))
        epistatic.append(np.sqrt(sum_sq_pred[i] / masks - p * p))
      if len(output) == 1:
        return (output[0], std[0], aleatoric[0], epistatic[0])
      else:
        return list(zip(output, std, aleatoric, epistatic))

  return KerasModel_uncertainty_wrapper
