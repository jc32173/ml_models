from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn_models.build_models.feat_prep_funcs import *


# SKLearn models

# ====================
# k Nearest Neighbours
# ====================

from sklearn.neighbors import KNeighborsRegressor

RmCorrVar_kNN = Pipeline([('RmConstDesc', RmConstDesc()),
                          ('RmCorrVar', RmCorrVar()),
                          ('scaler', StandardScaler()),
                          ('kNN', KNeighborsRegressor())])


# ================
# Ridge regression
# ================

from sklearn.linear_model import Ridge

Ridge = Ridge()


# ==========
# PLS models
# ==========

from sklearn.cross_decomposition import PLSRegression
#from sklearn_models.models import PLSDA

# PLS does X, y scaling as default:
PLS = PLSRegression(scale=True)

# Remove any constant descriptors:
PLS = Pipeline([('RmConstDesc', RmConstDesc()), 
                ('PLS', PLS)])


# ====================
# Random forest models
# ====================

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

RFR = RandomForestRegressor()
RFC = RandomForestClassifier()

# Remove any constant descriptors:
RFR = Pipeline([('RmConstDesc', RmConstDesc()), 
                ('RFR', RandomForestRegressor())])

# Remove correlated descriptors:
RmCorrVar_RFR = Pipeline([('RmConstDesc', RmConstDesc()), 
                          ('RmCorrVar', RmCorrVar()), 
                          ('RFR', RandomForestRegressor())])

# With RFE:
#RFR = Pipeline([('RmConstDesc', RmConstDesc()), ('RFR', RFR)])


# ==========
# SVM models
# ==========

from sklearn.svm import SVR, SVC

SVMR = SVR()
SVMC = SVC()

# SVM models with scaling:
SVMR = Pipeline([('RmConstDesc', RmConstDesc()),
                 ('scaler', StandardScaler()),
                 ('SVR', SVR())])
SVMC = Pipeline([('RmConstDesc', RmConstDesc()),
                 ('scaler', StandardScaler()),
                 ('SVC', SVC())])

RmCorrVar_SVMR = Pipeline([('RmConstDesc', RmConstDesc()),
                           ('RmCorrVar', RmCorrVar()),
                           ('scaler', StandardScaler()),
                           ('SVR', SVR())])
