from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold, RandomizedSearchCV, GridSearchCV

import numpy as np
import pandas as pd

# Random Forest

# Initial parameters for hyperparameter tuning:

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 20, stop = 300, num = 10)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 100, num = 10)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 6, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [2, 4, 6]

# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
init_param_grid = {'n_estimators'      : n_estimators,
                   'max_features'      : max_features,
                   'max_depth'         : max_depth,
                   'min_samples_split' : min_samples_split,
                   'min_samples_leaf'  : min_samples_leaf,
                   'bootstrap'         : bootstrap}

cv_tuning_stats = {}

def init_params(init_param_grid=init_param_grid):
    return init_param_grid

def tune_train_model(x_train, 
                     y_train, 
                     hyper_cv, 
                     init_param_grid=init_param_grid, 
                     model_type='regression', 
                     n_jobs=-1): #, cv_tuning_stats=cv_tuning_stats):

    ## RFE:
    #rfe_rand_seed = 34
    rf_rand_seed = 46
    #rf = RandomForestRegressor(n_estimators=100, random_state=rf_rand_seed + 1) #, n_estimators=100)
    #rfecv = RFECV(rf, cv=KFold(n_splits=hyper_cv, random_state=rfe_rand_seed), step=30, verbose=1, n_jobs=-1)
    #rfecv.fit(x_train, y_train)
    #x_train_rfe = rfecv.transform(x_train)

    # Run randomised search with k-fold cross-validation
    #rf = RandomForestRegressor(random_state=rf_rand_seed)
    if model_type == 'regression':
        estimator = RandomForestRegressor(random_state=rf_rand_seed)
    elif model_type == 'classifier':
        estimator = RandomForestClassifier(random_state=rf_rand_seed)
    rf_rand = RandomizedSearchCV(estimator = estimator,
                                 param_distributions = init_param_grid,
                                 n_iter = 10,
                                 random_state = 5678,
                                 cv = hyper_cv,
                                 refit = True,
                                 verbose = 1,
                                 n_jobs = n_jobs)

    #pipe = Pipeline([('rfecv', rfecv),
    #                 ('rf_rand', rf_rand)])

    #pipe.fit(x_train, y_train)

    #pipe_out = Pipeline([('rfecv', pipe['rfecv']), 
    #                     ('rf_rand', pipe['rf_rand'].best_estimator_)])

    #rf_rand.fit(x_train_rfe, y_train)
    rf_rand.fit(x_train, y_train)

    # Save stats from cv hyperparameter tuning rounds:
    #global cv_tuning_stats
    cv_tuning_stats = {}
    cv_tuning_stats['coarse'] = {'min'  : min(rf_rand.cv_results_['mean_test_score']), 
                                 'max'  : max(rf_rand.cv_results_['mean_test_score']), 
                                 'mean' : np.mean(rf_rand.cv_results_['mean_test_score']), 
                                 'var'  : np.var(rf_rand.cv_results_['mean_test_score'])**0.5}


    rf_grid = rf_rand
#    # Finer parameter grid:
#    rand_param = rf_rand.best_estimator_.get_params()['n_estimators']
#    n_estimators = [int(x) for x in np.linspace(start=max(1, rand_param-10), stop=rand_param+10, num=rand_param+5-max(1, rand_param-5)+1)]
#
#    max_features = [rf_rand.best_estimator_.get_params()['max_features']]
#
#    rand_param = rf_rand.best_estimator_.get_params()['max_depth']
#    if rand_param != None:
#        max_depth = [int(x) for x in np.linspace(max(1, rand_param-4), rand_param+4, num=rand_param+2-max(1, rand_param-2)+1)]
#    else:
#        max_depth = [rand_param]
#
#    rand_param = rf_rand.best_estimator_.get_params()['min_samples_split']
#    min_samples_split = [int(x) for x in np.linspace(max(2, rand_param-4), rand_param+4, num=rand_param+4-max(2, rand_param-4)+1)]
#
#    rand_param = rf_rand.best_estimator_.get_params()['min_samples_leaf']
#    min_samples_leaf = [int(x) for x in np.linspace(rand_param-1, rand_param+1, num=3)]
#
#    bootstrap = [rf_rand.best_estimator_.get_params()['bootstrap']]
#
#    finer_param_grid = {'n_estimators'      : n_estimators,
#                        'max_features'      : max_features,
#                        'max_depth'         : max_depth,
#                        'min_samples_split' : min_samples_split,
#                        'min_samples_leaf'  : min_samples_leaf,
#                        'bootstrap'         : bootstrap}
#
#    rf_grid = GridSearchCV(estimator = RandomForestRegressor(random_state=rf_rand_seed),
#                           param_grid = finer_param_grid,
#                           cv = hyper_cv,
#                           refit = True,
#                           verbose = 1,
#                           n_jobs = n_jobs)
#
#    rf_grid.fit(x_train, y_train)
#
#    cv_tuning_stats['fine'] = {'min'  : min(rf_grid.cv_results_['mean_test_score']), 
#                               'max'  : max(rf_grid.cv_results_['mean_test_score']), 
#                               'mean' : np.mean(rf_grid.cv_results_['mean_test_score']), 
#                               'var'  : np.var(rf_grid.cv_results_['mean_test_score'])**0.5}

    return rf_grid.best_estimator_, cv_tuning_stats

def get_trained_params(estimator):
    print(estimator.get_params())
    #return {'rfecv__n_features'          : estimator['rfecv'].n_features_,
    #        'rf_rand__n_estimators'      : estimator['rf_tuned'].get_params()['n_estimators'],
    #        'rf_rand__max_features'      : estimator['rf_tuned'].get_params()['max_features'],
    #        'rf_rand__max_depth'         : estimator['rf_tuned'].get_params()['max_depth'],
    #        'rf_rand__min_samples_split' : estimator['rf_tuned'].get_params()['min_samples_split'],
    #        'rf_rand__min_samples_leaf'  : estimator['rf_tuned'].get_params()['min_samples_leaf'],
    #        'rf_rand__bootstrap'         : estimator['rf_tuned'].get_params()['bootstrap'],
    #        'cv_tuning_stats'            : cv_tuning_stats}
    return {'rf_rand__n_estimators'      : estimator.get_params()['n_estimators'],
            'rf_rand__max_features'      : estimator.get_params()['max_features'],
            'rf_rand__max_depth'         : estimator.get_params()['max_depth'],
            'rf_rand__min_samples_split' : estimator.get_params()['min_samples_split'],
            'rf_rand__min_samples_leaf'  : estimator.get_params()['min_samples_leaf'],
            'rf_rand__bootstrap'         : estimator.get_params()['bootstrap']}
