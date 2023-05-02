"""
UNTESTED!
Class for doing multiclass PLS-DA.
"""


# March 2023


from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
#from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
#from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import accuracy_score, balanced_accuracy_score

import numpy as np
import pandas as pd
import pickle as pk


class PLSDA():

    def onehot_encode(fn):

    def unencode():

    def multiclass_plsda_score(estimator, x_valid, y_valid):
        # Assign predicted class as the class with maximum predicted value:
        y_valid_pred = estimator.predict(x_valid).argmax(axis=1)
        # Convert y data from one-hot encoding to integer class labels:
        y_valid = y_valid.to_numpy().argmax(axis=1)
        return balanced_accuracy_score(y_valid, y_valid_pred)

    def fit(X_train, y_train):
        y_train = pd.get_dummies(y_train)


        pipe = Pipeline([('scaler', StandardScaler()), 
                         ('pls', PLSRegression(scale=False))])

    init_param_grid = {'pls__n_components' : range(2, 20)}

    cv_split = StratifiedKFold(n_splits=hyper_cv, shuffle=True, random_state=1).split(x_train, y_train.to_numpy().argmax(axis=1))
    pls = GridSearchCV(estimator = pipe,
                       param_grid = init_param_grid,
                       scoring = multiclass_plsda_score,
                       cv = cv_split,
                       refit = True,
                       verbose = 1,
                       n_jobs = 1)

    ## Set up hyperparameter tuning using a random grid search over
    ## different combinations of hyperparameters:
    #pls = GridSearchCV(estimator = pipe,
    #                   param_grid = init_param_grid,
    #                   #n_iter = 10,
    #                   cv = hyper_cv,
    #                   refit = True,
    #                   verbose = 1,
    #                   n_jobs = 1)

    # Train RF model:
    pls.fit(x_train, y_train)

    #print(pls.best_estimator_.get_params())

    # Use trained RF model to predict y data for the test set:
    # Binary classification:
    #y_pred = np.array([0 if i < 0.5 else 1 for i in pls.predict(x_test)])
    # Multiclass classification:
    y_pred = pls.predict(x_test).argmax(axis=1)
    #y_pred = pd.get_dummies(y_pred)

    # Assess performace of model based on predictions:

    y_test = y_test.to_numpy().argmax(axis=1)

    ## Coefficient of determination
    #r2 = r2_score(y_test, y_pred)
    ## Root mean squared error
    #rmsd = mean_squared_error(y_test, y_pred)**0.5
    ## Bias
    #bias = np.mean(y_pred - y_test)
    ## Standard deviation of the error of prediction
    #sdep = np.mean(((y_pred - y_test) - np.mean(y_pred - y_test))**2)**0.5

    ## Save running sum of results:
    #r2_sum += r2
    #rmsd_sum += rmsd
    #bias_sum += bias
    #sdep_sum += sdep

    #print(y_test)
    #print(y_pred)

    accuracy.append(accuracy_score(y_test, y_pred))

    # Save individual predictions:
    # This may not be correct...
#    all_preds[n,test_idx] = [y_pred.columns[i] for i in y_pred.to_numpy().argmax(axis=1)]

# Average results over resamples:
#r2_av = r2_sum/mc_cv
#rmsd_av = rmsd_sum/mc_cv
#bias_av = bias_sum/mc_cv
#sdep_av = sdep_sum/mc_cv

# Write average results to a file:
#results_file = open(results_filename, 'w')
#results_file.write('r2: {:.3f}\n'.format(r2_av))
#results_file.write('rmsd: {:.3f}\n'.format(rmsd_av))
#results_file.write('bias: {:.3f}\n'.format(bias_av))
#results_file.write('sdep: {:.3f}\n'.format(sdep_av))
#results_file.close()

print(np.mean(accuracy))

# Save all individual predictions to file:
predictions_file = open(predictions_filename, 'w')
# Write header:
predictions_file.write(','.join([str(i) for i in y.index]) + '\n')
# Write individual predictions from each MC CV cycle:
for n in range(mc_cv):
    predictions_file.write(','.join([str(p) if not np.isnan(p) else '' for p in all_preds[n]]) + '\n')
predictions_file.close()
