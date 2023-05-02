from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

import numpy as np
import pandas as pd

# Ridge regression

# Initial parameters for hyperparameter tuning:
init_param_grid = {'rdg__alpha' :
                   [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000, 100000]}
                   #[x for x in np.arange(0, 100.1, 1)]}

def init_params(init_param_grid=init_param_grid):
    return init_param_grid

def tune_train_model(x_train, y_train, hyper_cv, init_param_grid=init_param_grid, n_jobs=-1, **kwargs):

    pipe = Pipeline([('scaler', StandardScaler()),
                     ('rdg', Ridge())])

    # Initial hyperparameter tuning:
    rdg_grid = GridSearchCV(estimator = pipe,
                            param_grid = init_param_grid,
                            cv = hyper_cv,
                            refit = True,
                            n_jobs = n_jobs,
                            verbose = 1)

    rdg_grid.fit(x_train, y_train)
    #print(rdg_grid.best_estimator_)

    cv_tuning_stats = {}    
    cv_tuning_stats['coarse'] = {'min'  : min(rdg_grid.cv_results_['mean_test_score']),
                                 'max'  : max(rdg_grid.cv_results_['mean_test_score']),
                                 'mean' : np.mean(rdg_grid.cv_results_['mean_test_score']),
                                 'sd'  : np.var(rdg_grid.cv_results_['mean_test_score'])**0.5}

    # Fine grained hyperparameter tuning:
    fine_param_grid = {'rdg__alpha' : 
                       [a*rdg_grid.best_estimator_.get_params()['rdg__alpha'] for a in range(1, 10)] +
                       [a*(rdg_grid.best_estimator_.get_params()['rdg__alpha']-1) for a in range(1, 10)]}

    pipe = Pipeline([('scaler', StandardScaler()),
                     ('rdg', Ridge())])

    rdg_fine_grid = GridSearchCV(estimator = pipe,
                                 param_grid = fine_param_grid,
                                 cv = hyper_cv,
                                 refit = True,
                                 n_jobs = n_jobs,
                                 verbose = 1)

    rdg_fine_grid.fit(x_train, y_train)

    cv_tuning_stats['fine'] = {'min'  : min(rdg_fine_grid.cv_results_['mean_test_score']),
                               'max'  : max(rdg_fine_grid.cv_results_['mean_test_score']),
                               'mean' : np.mean(rdg_fine_grid.cv_results_['mean_test_score']),
                               'var'  : np.var(rdg_fine_grid.cv_results_['mean_test_score'])**0.5}

    return rdg_fine_grid.best_estimator_, cv_tuning_stats

def get_trained_params(estimator, training_stats=None):
    return {'rdg__alpha' : estimator.get_params()['rdg__alpha']}
