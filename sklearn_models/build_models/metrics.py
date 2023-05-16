from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error, mean_absolute_error #, mean_poisson_deviance, mean_gamma_deviance
from scipy.stats import pearsonr, spearmanr
import sys
import numpy as np
import pandas as pd


# Functions to calculate scores:
regression_metrics = {'r2_score'   : lambda t, p : r2_score(t, p),
                      #'r2_score_m' : lambda t, p : 1 - (np.sum((t - p)**2)/np.sum((t - np.mean(t))**2)),
                      'rmsd'       : lambda t, p : (mean_squared_error(t, p))**0.5,
                      #'rmsd_m'     : lambda t, p : (np.mean((t - p)**2))**0.5,
                      'bias'       : lambda t, p : np.mean(p - t),
                      'sdep'       : lambda t, p : np.mean(((p - t) - np.mean(p - t))**2)**0.5,
                      'mae'        : lambda t, p : mean_absolute_error(t, p),
                      #'stddev'     : lambda t, _ : np.mean((t - np.mean(t))**2)**0.5,
                      'pearson'    : lambda t, p : pearsonr(t, p)[0],
                      #'pearsonp'   : lambda t, p : pearsonr(t, p)[1],
                      'spearman'   : lambda t, p : spearmanr(t, p)[0],
                      #'spearmanp'  : lambda t, p : spearmanr(t, p)[1],
                      #'explained_variance_score' : lambda t, p : explained_variance_score(t, p),
                      #'mean_poisson_dev' : lambda t, p : mean_poisson_deviance(p, t),
                      #'mean_gamma_dev' : lambda t, p : mean_gamma_deviance(p, t),
                      '_order' : ['r2_score', 'rmsd', 'bias', 'sdep', 'mae', 'pearson', 'spearman']
                     }

all_metrics = {'regression' : regression_metrics}

# Functions to calculate scores:
#regression_metrics = [('r2_score',  lambda t, p : r2_score(t, p)),
#                      #('r2_score_m', lambda t, p : 1 - (np.sum((t - p)**2)/np.sum((t - np.mean(t))**2)),)
#                      ('rmsd',      lambda t, p : (mean_squared_error(t, p))**0.5),
#                      #('rmsd_m',   lambda t, p : (np.mean((t - p)**2))**0.5,
#                      ('bias',      lambda t, p : np.mean(p - t)),
#                      ('sdep',      lambda t, p : np.mean(((p - t) - np.mean(p - t))**2)**0.5),
#                      ('mae',       lambda t, p : mean_absolute_error(t, p)),
#                      #('stddev',    lambda t, _ : np.mean((t - np.mean(t))**2)**0.5),
#                      ('pearson',   lambda t, p : pearsonr(t, p)[0]),
#                      #('pearsonp',  lambda t, p : pearsonr(t, p)[1]),
#                      ('spearman',  lambda t, p : spearmanr(t, p)[0]),
#                      #('spearmanp', lambda t, p : spearmanr(t, p)[1]),
#                      #('explained_variance_score', lambda t, p : explained_variance_score(t, p)),
#                      #('mean_poisson_dev', lambda t, p : mean_poisson_deviance(p, t)),
#                      #('mean_gamma_dev', lambda t, p : mean_gamma_deviance(p, t))
#                     ]

#all_metrics = {'regression' : [{'name' : metric_name, 'fn' : metric_fn} 
#                               for metric_name, metric_fn in regression_metrics]}

#all_metrics = {'regression' : pd.DataFrame(regression_metics, columns=['name', 'fn')}

def calc_scores(y_test, y_pred, all_metrics):

    model_scores = {}

    # Check y_test and y_pred are same shape:
    # predict from tensor flow gives array of arrays so need to squeeze()
    if y_test.shape != y_pred.shape:
        raise ValueError('Shape of y_pred does not match y_test.')

    # Calculate scores:
    for metric_name, metric_fn in all_metrics.items():
        model_scores[metric_name] = metric_fn(y_test, y_pred)
    return model_scores
