from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score, fbeta_score, cohen_kappa_score, matthews_corrcoef, jaccard_score, roc_auc_score, precision_score, recall_score #, mean_poisson_deviance, mean_gamma_deviance
from scipy.stats import pearsonr, spearmanr
import sys
import numpy as np
import pandas as pd

def calc_model_scores(y_test, y_pred, model_scores, print_scores=True):

        # Check y_test and y_pred are same shape:
        # predict from tensor flow gives array of arrays so need to squeeze()
        if y_test.shape != y_pred.shape:
                sys.exit('ERROR in calc_model_scores: Shape of y_pred does not match y_test.')

        print('Number of predictions:', len(y_pred))

        n_corr = np.zeros(3)
        sens = np.zeros(3)
        spec = np.zeros(3)
        ppv = np.zeros(3)
        npv = np.zeros(3)

        def c_mat_stats(y_test, y_pred):
                c_mat = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
                for i in range(c_mat.shape[0]):
        
                        n_corr[i] = np.sum((np.array(y_test) == i) & (np.array(y_pred) == i))
        
                        TP = c_mat[i,i]
                        FP = np.sum([elem for elem_i, elem in enumerate(c_mat[:,i]) if (elem_i != i)])
                        FN = np.sum([elem for elem_i, elem in enumerate(c_mat[i]) if (elem_i != i)])
                        TN = np.sum([elem for row_i, row in enumerate(c_mat) for elem_i, elem in enumerate(row) if (elem_i != i and row_i != i)])
        
                        sens[i] = TP/(TP+FN)
                        spec[i] = TN/(TN+FP)
                        ppv[i]  = TP/(TP+FP)
                        npv[i]  = TN/(TN+FN)

                return sens, spec, ppv, npv, n_corr

        def run_roc_auc(t, p, multi_class='ovo'):
            p_prob = np.zeros((len(p), 3))
            for i, cls in enumerate(p):
                p_prob[i,int(cls)] = 1
            print(t)
            print(p_prob)
            return roc_auc_score(t, p_prob, multi_class=multi_class) #, average='macro') #, multi_class='ovo') #) #, multi_class='ovr')
#multi_class

        # Functions to calculate scores:
        calc_scores = {#'confusion_mat_stats' : lambda t, p : c_mat_stats(t, p),
                       'sens'          : lambda t, p : c_mat_stats(t, p)[0],
                       'spec'          : lambda t, p : c_mat_stats(t, p)[1],
                       'ppv'           : lambda t, p : c_mat_stats(t, p)[2],
                       'npv'           : lambda t, p : c_mat_stats(t, p)[3],
                       'n_corr'        : lambda t, p : c_mat_stats(t, p)[4],
                       'accuracy'      : lambda t, p : accuracy_score(t, p),
                       'accuracy_m'    : lambda t, p : np.sum(t == p)/len(t),
                       'balanced_accuracy' : lambda t, p : balanced_accuracy_score(t, p),
                       'f1_score'      : lambda t, p : f1_score(t, p, average='micro'),
                       #'fbeta_score'   : lambda t, p : fbeta_score(t, p),
                       'cohen_k'       : lambda t, p : cohen_kappa_score(t, p),
                       'matthews_coef' : lambda t, p : matthews_corrcoef(t, p),
                       'jaccard_score' : lambda t, p : jaccard_score(t, p, average='micro'),
                       'roc_auc_score_ovo' : lambda t, p : run_roc_auc(t, p, 'ovo'),
                       'roc_auc_score_ovr' : lambda t, p : run_roc_auc(t, p, 'ovr'),
                       'precision_ppv' : lambda t, p : precision_score(t, p, average=None),
                       'recall_sens'   : lambda t, p : recall_score(t, p, average=None)
                      }

        # Calculate scores:
        for metric in model_scores.keys():
                score = calc_scores[metric](y_test, y_pred)
                #print(score)
                if print_scores == True:
                        print(metric + ": " + str(score))
                model_scores[metric].append(score)

        return model_scores
