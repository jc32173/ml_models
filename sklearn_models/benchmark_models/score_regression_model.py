from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error, mean_absolute_error #, mean_poisson_deviance, mean_gamma_deviance
from scipy.stats import pearsonr, spearmanr
import sys
import numpy as np
import pandas as pd

def calc_model_scores(y_test, y_pred, model_scores, print_scores=True):

        # Check y_test and y_pred are same shape:
        # predict from tensor flow gives array of arrays so need to squeeze()
        if y_test.shape != y_pred.shape:
                sys.exit('ERROR in calc_model_scores: Shape of y_pred does not match y_test.')

        if print_scores:
	        print('Number of predictions:', len(y_pred))
	# Functions to calculate scores:
        calc_scores = {'r2_score'   : lambda t, p : r2_score(t, p),
		       'r2_score_m' : lambda t, p : 1 - (np.sum((t - p)**2)/np.sum((t - np.mean(t))**2)),
                       'rmsd'       : lambda t, p : (mean_squared_error(t, p))**0.5,
                       'rmsd_m'     : lambda t, p : (np.mean((t - p)**2))**0.5,
                       'bias'       : lambda t, p : np.mean(p - t),
                       'rand_error' : lambda t, p : np.mean(((p - t) - np.mean(p - t))**2)**0.5,
                       'testset_stddev' : lambda t, _ : np.mean((t - np.mean(t))**2)**0.5,
                       'pearsonr'   : lambda t, p : pearsonr(t, p)[0],
                       'pearsonp'   : lambda t, p : pearsonr(t, p)[1],
                       'spearmanr'  : lambda t, p : spearmanr(t, p)[0],
                       'spearmanp'  : lambda t, p : spearmanr(t, p)[1],
                       'mean_abs_error' : lambda t, p : mean_absolute_error(t, p),
                       'explained_variance_score' : lambda t, p : explained_variance_score(t, p),
                       #'mean_poisson_dev' : lambda t, p : mean_poisson_deviance(p, t),
                       #'mean_gamma_dev' : lambda t, p : mean_gamma_deviance(p, t)
                      }

        # Calculate scores:
        for metric in model_scores.keys():
                score = calc_scores[metric](y_test, y_pred)
                if print_scores == True:
                        print(metric + ": " + str(score))
                model_scores[metric].append(score)
        return model_scores
