from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr, kendalltau
import deepchem as dc
import numpy as np
import math

# Define deepchem metrics:

# Regression:

r2 = dc.metrics.Metric(metric=r2_score, name='r2')
rmsd = dc.metrics.Metric(metric=dc.metrics.rms_score, name='rmsd')
mse = dc.metrics.Metric(metric=mean_squared_error, name='mse')

bias = dc.metrics.Metric(metric=lambda t, p: np.mean(p - t), name='bias', mode='regression')
sdep = dc.metrics.Metric(metric=lambda t, p: math.sqrt(np.mean(((p - t) - np.mean(p - t))**2)), name='sdep', mode='regression')

mae = dc.metrics.Metric(metric=mean_absolute_error, name='mae')
pearson = dc.metrics.Metric(metric=lambda t, p: pearsonr(t, p)[0], name='pearson', mode='regression')
spearman = dc.metrics.Metric(metric=lambda t, p: spearmanr(t, p)[0], name='spearman', mode='regression')
kendall = dc.metrics.Metric(metric=lambda t, p: kendalltau(t, p)[0], name='kendall', mode='regression')

# Don't add to list of metrics, but use with evaluate
# so that transforms can be applied:
#y_stddev = dc.metrics.Metric(metric=lambda t, p: np.std(t), name='y_stddev', mode='regression')

# Maybe better to add this as a complete function, since don't 
# need to use model.evaluate, probably doesn't need to be a metric 
# either, just calculate from the dataset
def calc_stddev(y, 
                transformers=[]):
    y = dc.trans.undo_transforms(y, transformers)
    stddev = np.std(y)
    return stddev

# Classification:

accuracy = dc.metrics.Metric(metric=dc.metrics.accuracy_score, name='accuracy', mode='classification') #, n_classes=10)
accuracy.classification_handling_mode = "threshold-one-hot"

#balanced_accuracy = dc.metrics.Metric(metric=dc.metrics.balanced_accuracy_score, name='balanced_accuracy', mode='classification')
#balanced_accuracy.classification_handling_mode = "threshold-one-hot"

#matthews_corrcoef = dc.metrics.Metric(metric=dc.metrics.matthews_corrcoef, name='matthews_corrcoef', mode='classification')
#matthews_corrcoef.classification_handling_mode = "threshold-one-hot"

recall = dc.metrics.Metric(metric=lambda t, p : dc.metrics.recall_score(t, p, average='micro'), name='recall', mode='classification')
recall.classification_handling_mode = "threshold-one-hot"

precision = dc.metrics.Metric(metric=lambda t, p : dc.metrics.precision_score(t, p, average='micro'), name='precision', mode='classification')
precision.classification_handling_mode = "threshold-one-hot"


all_metrics = {'regression' : {'r2' : r2,
                               'rmsd' : rmsd, 
                               'mse' : mse,
                               'bias' : bias,
                               'sdep' : sdep,
                               'mae' : mae,
                               'pearson' : pearson,
                               'spearman' : spearman, 
                               'kendall' : kendall,
                               '_order' : ['r2', 'rmsd', 'mse', 'bias', 'sdep', 
                                           'mae', 'pearson', 'spearman', 'kendall']}, 
               'classification' : {'accuracy' : accuracy,
                                  #'balanced_accuracy' : balanced_accuracy,
                                  #'matthews_corrcoef' : matthews_corrcoef,
                                   'recall' : recall,
                                   'precision' : precision,
                                   '_order' : ['accuracy', #'balanced_accuracy', 
                                              #'matthews_corrcoef', 
                                               'recall', 'precision'
                                               ]}}
