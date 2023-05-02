from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np

# Remove constant descriptors
def rm_const_desc(x_train, x_test=None):
    """
    Remove any descriptors which are constant for all samples.
    """
#    if not isinstance(x_test, list):
#        x_test = [x_test]

    del_descs = []
    for col in x_train.columns:
        if np.all([x_train[col].iloc[0] == x_train[col]]) == True:
            del_descs.append(col)
    x_train = x_train.drop(columns=del_descs)
    if x_test is None:
        return x_train, len(del_descs)
#    if x_test[0] is [None]:
#        return x_train, len(del_descs)
    else:
        x_test = x_test.drop(columns=del_descs)
        return x_train, x_test, len(del_descs)
#    elif len(x_test) == 1:
#        x_test = x_test[0].drop(columns=del_descs)
#        return x_train, x_test, len(del_descs)
#    else:
#        x_test_scaled = []
#        for x_t in x_test:
#            x_test_scaled.append(x_t.drop(columns=del_descs))
#        return x_train, x_test_scaled, len(del_descs)

# Scale all x data:
def scale_x(x_train, x_test=None):
    """Centre and scale x data"""
    #if not isinstance(x_test, list):
    #    x_test = [x_test]

    scaler = StandardScaler()
    x_train = pd.DataFrame(data=scaler.fit_transform(x_train),
                           columns=x_train.columns,
                           index=x_train.index)
    #if x_test[0] is None:
    if x_test is None:
        return x_train
    else:
        x_test = pd.DataFrame(data=scaler.fit_transform(x_test),
                              columns=x_test.columns,
                              index=x_test.index)
        return x_train, x_test

#    elif len(x_test) == 1:
#        x_test = pd.DataFrame(data=scaler.fit_transform(x_test[0]),
#                              columns=x_test[0].columns,
#                              index=x_test[0].index)
#        return x_train, x_test
#    else:
#        x_t_ls = []
#        for x_t in x_test:
#            x_t_ls.append(pd.DataFrame(data=scaler.fit_transform(x_test),
#                                       columns=x_test.columns,
#                                       index=x_test.index))
#        return x_train, x_t_ls

# Remove descriptors which have a high correlation:
def rm_corr_var(corr_cut, x_train, x_test=None):
    """
    Remove correlated descriptors
    """
#    if not isinstance(x_test, list):
#        x_test = [x_test]

    # If DataFrame:
    x_corr = x_train.corr() - np.diag(np.ones(x_train.shape[1]))
    x_corr = np.abs(x_corr.to_numpy())

    # If numpy:
    #x_corr = np.abs(np.corrcoef(x_train, rowvar=False)) - np.diag(np.ones(x_train.shape[1]))

    x_corr_sort1d = x_corr.argsort(axis=None)[::-1]
    x_corr_sort2d = np.vstack((np.unravel_index(x_corr_sort1d, x_corr.shape))).T

    del_idxs = []
    for i, j in x_corr_sort2d:
        if x_corr[i, j] < corr_cut:
            break
        if i not in del_idxs and j not in del_idxs:
            rm_col = max(i, j)
            del_idxs.append(rm_col)

    n_corr_var = len(del_idxs)
    print('Removing {} features with corelation > {}'.format(n_corr_var, corr_cut))

    x_train.drop(x_train.iloc[:,del_idxs], axis=1, inplace=True)
    if x_test is None:
        return x_train, n_corr_var
    else:
        x_test.drop(x_test.iloc[:,del_idxs], axis=1, inplace=True)
        return x_train, x_test, n_corr_var

# Run RFE here so that results can be used in multiple models:
def rfe_select(x_train, y_train, x_test=None, c_train=None, rfe_max=None, rfe_min=1, rfe_cv=5, rfe_strat=False, n_jobs=-1):
    rfe_rand_seed = 34
    rf_rand_seed = 56

    # Alternatively use pipe?
    #pipe = Pipeline(['rfe' : rfe])

    if rfe_max is not None and rfe_max < x_train.shape[1]:
        rf = RandomForestRegressor(n_estimators=100, random_state=rf_rand_seed)
        rfe = RFE(rf, n_features_to_select=rfe_max, step=1, verbose=1)
        rfe.fit(x_train, y_train)
        x_train = pd.DataFrame(data=rfe.transform(x_train),
                               columns=x_train.columns[rfe.support_],
                               index=x_train.index)
        if x_test is not None:
            x_test = pd.DataFrame(data=rfe.transform(x_test),
                                  columns=x_test.columns[rfe.support_],
                                  index=x_test.index)

    if rfe_strat == True:
        rfe_cv_iter = StratifiedKFold(n_splits=rfe_cv,
                                      random_state=rfe_rand_seed,
                                      shuffle=True
                                     )
    else:
        rfe_cv_iter = KFold(n_splits=rfe_cv,
                            random_state=rfe_rand_seed,
                            shuffle=True
                           )
    rfe_cv_iter = [[list(i), list(j)] for i, j in rfe_cv_iter.split(x_train, c_train)]

    rf = RandomForestRegressor(n_estimators=100, random_state=rf_rand_seed)
    rfecv = RFECV(rf, min_features_to_select=rfe_min, cv=rfe_cv_iter, step=1, verbose=1, n_jobs=n_jobs)
    rfecv.fit(x_train, y_train)

    x_train = pd.DataFrame(data=rfecv.transform(x_train),
                           columns=x_train.columns[rfecv.support_],
                           index=x_train.index)
    if x_test is None:
        return x_train
    else:
        x_test = pd.DataFrame(data=rfecv.transform(x_test),
                              columns=x_test.columns[rfecv.support_],
                              index=x_test.index)
        return x_train, x_test

#def write_results():
#    df_out = 
#
#    for col in header:
#        for write(col)
#
#    # Save models, doesn't work if cv is a generator, have to turn it into a list (above)
#    if rfe_max is not None:
#        pk.dump([rfe, rfecv], open('RFE_models_'+str(n)+'.pk', 'wb'))
#    else:
#        pk.dump([rfecv], open('RFE_models_'+str(n)+'.pk', 'wb'))
#
#    x_train = pd.DataFrame(data=rfecv.transform(x_train),
#                           columns=x_train.columns[rfecv.support_],
#                           index=x_train.index)
#    if x_test is None:
#        return x_train
#    else:
#        x_test = pd.DataFrame(data=rfecv.transform(x_test),
#                              columns=x_test.columns[rfecv.support_],
#                              index=x_test.index)
#        return x_train, x_test
#
#
