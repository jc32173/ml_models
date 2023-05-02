from sklearn import __version__ as sklearn_ver
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit #, KFold, StratifiedKFold # train_test_split
#from sklearn.preprocessing import StandardScaler
#from sklearn.feature_selection import RFE, RFECV
#from sklearn.ensemble import RandomForestRegressor

import sys, os
import importlib
import numpy as np
import pandas as pd
import pickle as pk
from datetime import datetime

sys.path.insert(0, '/users/xpb20111/scripts/')
sys.path.insert(0, os.getcwd())

import score_regression_model as sc
from prep_funcs import rm_const_desc, scale_x, rm_corr_var, rfe_select
from calc_desc import calc_desc_df

usage = """
./Build_validate_model.py [-h, --help]
                          [-f data file]
                          [-calc_descs
                              -smi_col Name of SMILES column
                              -y_col Name of y column
                              -c_col Name of c column
                              -g_col Name of g column
                              -tauto Do RDKit tautomer canonicalisation
                              -pH Convert SMILES to pH
                              -pH_model OpenBabel or OpenEye
                              -desc Name of descriptor set]
                          [-col_order order of columns in data file]
                          [-strat fit based on stratified splits]
                         # [-score_strat calculate stratified scores]
                          [-m ML models]
                          [-prev_feat_select file containing previously selected features]
                          [-corr_cut Remove correlated features]
                          [-rfe bool]
                          [-rfe_max Maximum number of components]
                          [-rfe_min Minimum number of components]
                          [-rfe_strat]
                          [-rfe_cv Number of CV folds for RFE]
                         # [-o scores files]
                          [-mc_cv number of MC CV cycles]
                          [-hyper_cv folds of CV for hyperparameter tuning/epochs of NN]
                          [-group groups to be preserved in train/test split
                                  (must be in same stratified class)]
                          [-refit]
                          [-rescore?]
                          [-dlim delimiter for output file]
                          [-n_jobs number of processors to use for training]
"""

# model info:
# 1D dataframe to allow for different dtypes
di = {} #pd.DataFrame()


# Help:
if '-h' in sys.argv or '--help' in sys.argv or len(sys.argv) < 2:
    sys.exit(usage)

# Datafile:
if '-f' in sys.argv:
    di['data_file'] = sys.argv[((sys.argv).index('-f'))+1]
    data_file = sys.argv[((sys.argv).index('-f'))+1]
else:
    sys.exit('ERROR: No data file given (-f).')

# Order of columns in datafile:
# (If descriptors have been pre-calculated)
if '-col_order' in sys.argv:
    col_order = []
    for col in (sys.argv[((sys.argv).index('-col_order'))+1:]):
        if col[0] == '-':
            break
        col_order.append(col)
    if 'x' not in col_order or 'y' not in col_order:
        sys.exit('ERROR: -col values must include x and y')
    elif len([i for i in col_order if i not in ['x', 'y', 'g', 'c', '_']]) > 0:
        sys.exit('ERROR: Unknown -col value, must be in [x, y, g, c, _]')
else:
    col_order = ['x', 'y']

calc_descs = False
if '-calc_descs' in sys.argv:
    calc_descs = True
    if '-smi_col' in sys.argv:
        di['smi_col'] = (sys.argv[((sys.argv).index('-smi_col'))+1])
        smi_col = (sys.argv[((sys.argv).index('-smi_col'))+1])
    else:
        smi_col = None
    if '-y_col' in sys.argv:
        di['y_col'] = (sys.argv[((sys.argv).index('-y_col'))+1])
        y_col = (sys.argv[((sys.argv).index('-y_col'))+1])
    else:
        y_col = None
    if '-c_col' in sys.argv:
        di['c_col'] = (sys.argv[((sys.argv).index('-c_col'))+1])
        c_col = (sys.argv[((sys.argv).index('-c_col'))+1])
    else:
        c_col = None
    if '-g_col' in sys.argv:
        di['g_col'] = (sys.argv[((sys.argv).index('-g_col'))+1])
        g_col = (sys.argv[((sys.argv).index('-g_col'))+1])
    else:
        g_col = None

    if '-desc' in sys.argv:
        di['desc'] = (sys.argv[((sys.argv).index('-desc'))+1])
        desc = (sys.argv[((sys.argv).index('-desc'))+1])
        if desc not in ['RDKit2D', 'Mordred2D']:
            sys.exit('ERROR: Descriptors not implemented')

    if '-tauto' in sys.argv:
        di['tauto'] = 'RDKit'
        tauto = 'RDKit'
    else:
        tauto = None

    if '-pH' in sys.argv:
        ph = float(sys.argv[((sys.argv).index('-pH'))+1])
        di['ph'] = ph
    elif '-ph' in sys.argv:
        ph = float(sys.argv[((sys.argv).index('-ph'))+1])
        di['ph'] = ph
    else:
        ph = None
        di['ph'] = ph
    if '-pH_model' in sys.argv:
        phmodel = sys.argv[((sys.argv).index('-pH_model'))+1]
        di['ph_model'] = phmodel
    else:
        phmodel = None

#if '-o' in sys.argv:
#    scores_file = {}
#    for mod_i, output_file in enumerate(sys.argv[((sys.argv).index('-o'))+1:]):
#        if output_file[0] == '-':
#            break
#        scores_file[ml_model_ls[mod_i]] = output_file
#else:
#    sys.exit('ERROR: No output file given (-o).')

# Import ML models:
if '-m' in sys.argv:
    ml_models = {}
    ml_model_order = []
    for ml_model in (sys.argv[((sys.argv).index('-m'))+1:]):
        if ml_model[0] == '-':
            break
        ml_models[ml_model] = importlib.import_module(ml_model)
        ml_model_order.append(ml_model)
else:
    sys.exit('ERROR: No ML model module given (-m).')

prev_feat_select = None
if '-prev_feat_select' in sys.argv:
    rfe_results = pk.load(open(sys.argv[((sys.argv).index('-prev_feat_select'))+1], 'rb'))
    di.update(prev_feat_select)
else:
    rfe_results_file = 'RFE_results.pk'

# Use RFE in some models:
#rfe_mask = [None]
if '-rfe' in sys.argv:
    rfe_mask = []
    for rfe_bool in (sys.argv[((sys.argv).index('-rfe'))+1:]):
        if rfe_bool[0] == '-':
            break
        elif rfe_bool == 'True':
            rfe_mask.append(True)
        elif rfe_bool == 'False':
            rfe_mask.append(False)
        else:
            sys.exit('ERROR: -rfe must be sequence containing True and False only')
    rfe_mask = np.array(rfe_mask)
    if len(rfe_mask) != len(ml_models):
        sys.exit('ERROR: -rfe sequence must be the same length as list of ML models')
else:
    rfe_mask = np.array([False for _ in range(len(ml_model_order))])

# Set maximum number of components to select in RFE:
if '-rfe_max' in sys.argv:
    #rfei[('rfe', 'rfe_max')] = int(sys.argv[((sys.argv).index('-rfe_max'))+1])
    rfe_max = int(sys.argv[((sys.argv).index('-rfe_max'))+1])
else:
    #rfei[('rfe', 'rfe_max')] = None
    rfe_max = None

# Set minimum number of components to select in RFE:
if '-rfe_min' in sys.argv:
    #rfei[('rfe', 'rfe_min')] = int(sys.argv[((sys.argv).index('-rfe_min'))+1])
    rfe_min = int(sys.argv[((sys.argv).index('-rfe_min'))+1])
else:
    #rfei[('rfe', 'rfe_min')] = 1
    rfe_min = 1

# Do stratified CV in RFECV:
if '-rfe_strat' in sys.argv:
    #rfei[('rfe', 'rfe_strat')] = True
    rfe_strat = True
else:
   # rfei[('rfe', 'rfe_strat')] = False
    rfe_strat = False

# Number of folds used for RFECV:
if '-rfe_cv' in sys.argv:
    #rfei[('rfe', 'rfe_cv')] = int(sys.argv[((sys.argv).index('-rfe_cv'))+1])
    rfe_cv = int(sys.argv[((sys.argv).index('-rfe_cv'))+1])
else:
    #rfei[('rfe', 'rfe_cv')] = 5
    rfe_cv = 5

if '-corr_cut' in sys.argv:
    di[('corr_cut')] = float(sys.argv[((sys.argv).index('-corr_cut'))+1])
    corr_cut = float(sys.argv[((sys.argv).index('-corr_cut'))+1])
else:
    di[('corr_cut')] = None
    corr_cut = None

# Delimiter for results file:
if '-dlim' in sys.argv:
    dlim = sys.argv[((sys.argv).index('-dlim'))+1]
else:
    dlim = ';'

# Fraction of data used for test set:
if '-test_frac' in sys.argv:
    di['test_frac'] = sys.argv[((sys.argv).index('-dlim'))+1]
    test_frac = sys.argv[((sys.argv).index('-dlim'))+1]
else:
    di['test_frac'] = 0.3
    test_frac = 0.3

# Number of MC CV rounds:
if '-mc_cv' in sys.argv:
    di['mc_cv'] = int(sys.argv[((sys.argv).index('-mc_cv'))+1])
    mc_cv = int(sys.argv[((sys.argv).index('-mc_cv'))+1])
else:
    di['mc_cv'] = 20
    mc_cv = 20

# Number of CV folds for hyperparameter tuning:
if '-hyper_cv' in sys.argv:
    di['hyper_cv'] = int(sys.argv[((sys.argv).index('-hyper_cv'))+1])
    hyper_cv = int(sys.argv[((sys.argv).index('-hyper_cv'))+1])
else:
    di['hyper_cv'] = 10
    hyper_cv = 10

# Number of processors:
if '-n_jobs' in sys.argv:
    di['n_jobs'] = int(sys.argv[((sys.argv).index('-n_jobs'))+1])
    n_jobs = int(sys.argv[((sys.argv).index('-n_jobs'))+1])
else:
    di['n_jobs'] = -1
    n_jobs = -1

if '-strat' in sys.argv:
    di['strat'] = True
    strat = True
else:
    di['strat'] = False
    strat = False

if '-group' in sys.argv:
    di['group'] = True
    group = True
else:
    di['group'] = False
    group = False

if '-prpty' in sys.argv:
    di['prpty'] = sys.argv[((sys.argv).index('-prpty'))+1]
    prpty = sys.argv[((sys.argv).index('-prpty'))+1]
else:
    di['prpty'] = ''
    prpty = ''

data_file_header = False
if '-dataset' in sys.argv:
    di['dataset'] = sys.argv[((sys.argv).index('-dataset'))+1]
    dataset = sys.argv[((sys.argv).index('-dataset'))+1]
    #if di['dataset'] == 'InFile':
    if dataset == 'InFile':
        data_file_header = True
else:
    di['dataset'] = ''
    dataset = ''

if '-desc' in sys.argv:
    di['desc_set'] = sys.argv[((sys.argv).index('-desc'))+1]
    desc_set = sys.argv[((sys.argv).index('-desc'))+1]
else:
    di['desc_set'] = ''
    desc_set = ''

if '-ext_test' in sys.argv:
    ext_ls = []
    for ext in (sys.argv[((sys.argv).index('-ext_test'))+1:]):
        if ext[0] == '-':
            break
        else:
            ext_ls.append(ext)
    ext_ls = np.array(ext_ls)
    ext_ls = ext_ls.reshape(len(ext_ls)//2, 2)

#    for ext_datafile in (sys.argv[((sys.argv).index('-ext_test'))+1:]):
#        if ext_datafile[0] == '-':
#            break
#        else:
#            ext_testset_files.append(ext_datafile)


dataset_notes = ''


# Metrics to save:
metric_ls = ['r2_score',
#             'r2_score_m',
             'rmsd',
#             'rmsd_m',
             'bias',
             'rand_error',
             'testset_stddev',
             'pearsonr',
#             'pearsonp',
             'spearmanr',
#             'spearmanp',
             'mean_abs_error',
#             'explained_variance_score',
#             'mean_poisson_dev',
#             'mean_gamma_dev'
             ]


# Read dataset 
# ------------

# Read column names and calculate descriptors internally:

print('Reading data...')
if calc_descs:
    print('Calculating descriptors...')
    df = pd.read_csv(data_file)

    df_desc = calc_desc_df(df[smi_col], tauto, ph, phmodel)

    #df_desc[['Warnings']] = df_desc.apply()

    # Save descriptors:
    df_desc.to_csv('DESC.csv')

    x = df_desc.drop(['Processed_SMILES', 'Warnings'], axis=1)

    df.set_index(smi_col, inplace=True, verify_integrity=True)
    y = df[y_col]

    cls_labels = None
    if c_col is not None:
        c = df[c_col]
        cls_labels = list(c.unique()) #.to_list()
    if g_col is not None:
        g = df[g_col]

    # Check no NaN in x and remove if found:
    if x.isnull().values.any():
        print('WARNING: Initial set of descriptors contained NaN values:')
        print('Descriptor\tNumber of molecules with NaN values')
        for d_nan in x.columns[x.isnull().any()]:
            print(d_nan+'\t'+sum(x[d_nan].isna()))
        print('These will be removed')
        x = x.drop(x.columns[x.isnull().any()], axis=1, inplace=True)

else:
    # Easier to save data as a dictionary?  Then could read as:
    #x = all_data['x']
    #y = all_data['y']
    #c = all_data.get('c')
    #g = all_data.get('g')
    #notes = all_data.get('notes')

    #all_data = pk.load(open(di['data_file'], "rb"))
    all_data = pk.load(open(data_file, "rb"))

    if data_file_header == True:
        di['dataset'] = all_data[0]
        all_data = all_data[1]

    x = all_data[col_order.index('x')]
    y = all_data[col_order.index('y')]
    cls_labels = None
    if 'c' in col_order:
        c = all_data[col_order.index('c')]
        cls_labels = list(set(c))
        cls_labels.sort()
    #elif di['strat'] == True:
    elif strat == True:
        sys.exit('ERROR: c column not given to -col')
    #if di['group'] == True:
    if group == True:
        if 'g' not in cols:
            sys.exit('ERROR: g column not given to -col')
        g = all_data[cols.index('g')]

print('Dataset shape: X: {}, y: {}'.format(str(x.shape), str(y.shape)))


# Objects to store feature selection from each run:
rfe_results = {#'data_file' : di['data_file'],
               'data_file' : data_file,
               #'test_frac' : di['test_frac'],
               'test_frac' : test_frac,
               #'corr_cut' : di['corr_cut'],
               'corr_cut' : corr_cut,
               #'strat' : di['strat'],
               'strat' : strat,
               #'n_features' : [[] for _ in range(mc_cv)],
               'features' : [[] for _ in range(mc_cv)],
               'test_idx' : [[] for _ in range(mc_cv)],
#              'ranking' : [[] for _ in range(mc_cv)],
#              'grid_scores' : [[] for _ in range(mc_cv)],
#              'preselect_features' : [[] for _ in range(mc_cv)]
                      }

#feat_select_results.update(rfei)

# Dictionaries to store results in:
tuned_params = {}
cv_tuning_stats = {}
model_scores = {}
all_preds = {}

# Files to save results to:

# Stats averaged over all MC CV cycles:
av_stats_outfile = {}
# Stats from each individual MC CV cycle:
mc_cv_stats_outfile = {}
# Save individual predictions:
preds_outfile = {}

for ml_model in ml_model_order:

    # Tuned hyperparameters:
    tuned_params[ml_model] = [{} for _ in range(mc_cv)]
    cv_tuning_stats[ml_model] = [{} for _ in range(mc_cv)]
    model_scores[ml_model] = {}
    all_preds[ml_model] = {}

    for data_split in ['train', 'test']:
        model_scores[ml_model][data_split] = {'Total' : {}}
        for metric in metric_ls:
            model_scores[ml_model][data_split]['Total'][metric] = []
        if c_col is not None or 'c' in col_order:
            for cls in cls_labels:
                model_scores[ml_model][data_split][cls] = {}
                for metric in metric_ls:
                    model_scores[ml_model][data_split][cls][metric] = []

        # Different ways of storing predictions:
        #all_preds[ml_model][data_split] = [() for _ in range(mc_cv)]
        #all_preds[ml_model][data_split] = np.zeros((mc_cv))
        all_preds[ml_model][data_split] = np.empty((mc_cv, x.shape[0]), dtype=float)
        all_preds[ml_model][data_split][:] = np.nan

    # Output filenames:
    av_stats_outfile[ml_model] = ml_model+'_av_stats.dat'
    mc_cv_stats_outfile[ml_model] = ml_model+'_indiv_resample_stats.dat'
    preds_outfile[ml_model] = {}
    for data_split in ['train', 'test']:
        preds_outfile[ml_model][data_split] = ml_model+'_'+data_split+'_preds.dat'

    # rfe_outfile[][] = 'rfe_mc_cv_results.pk'

# Write headers for file mc_cv_stats_outfile since this is open and written to during training:
# Use ','.join?
for ml_model in ml_model_order:
    mc_cv_stats_outfile[ml_model] = open(mc_cv_stats_outfile[ml_model], 'w')
    out = mc_cv_stats_outfile[ml_model]
    out.write('MC_CV_cycle')

    #for metric in metric_ls:
        #out.write("{}{}".format(dlim, metric+'_test'))
    out.write(dlim.join([metric+'_test' for metric in metric_ls]))

    out.write("{}{}".format(dlim, 'n_const_descs'))
    out.write("{}{}".format(dlim, 'n_corr_variables'))
    if True in rfe_mask:
        out.write("{}{}{}{}".format(dlim, 'rfe_n_features',
                                    dlim, 'rfe_features')
                                    #dlim, rfe_results['ranking'][n],
                                    #dlim, rfe_results['grid_scores'][n],
                                    #dlim, rfe_results['preselect_features'][n])
                                   )

    # Performance metrics on training set:
    for metric in metric_ls:
        out.write("{}{}".format(dlim, metric+'_training'))

    out.write("{}{}".format(dlim, 'tuned_params'))
    out.write("{}{}".format(dlim, 'cv_tuning_stats'))
    out.write("\n")
    # Make sure file is written at the end of each CV cycle:
    out.flush()

    # Predictions file:
    for data_split in ['train', 'test']:
        preds_outfile[ml_model][data_split] = open(preds_outfile[ml_model][data_split], 'w')
        preds_outfile[ml_model][data_split].write(dlim.join([str(idx) for idx in y.index]))
        preds_outfile[ml_model][data_split].write('\n')

def calc_stddev(a):
    if len(a) > 0:
        return (np.sum((a - np.mean(a))**2)/len(a))**0.5
    else:
        return np.nan

def calc_stderr(a):
    if len(a) > 1:
        return ((np.sum((a - np.mean(a))**2)/(len(a) - 1))**0.5)/(len(a)**0.5)
    else:
        return np.nan

def print_header(title, surr='-'):
    print(surr*len(title)+'\n'+title+'\n'+surr*len(title))

# def train_eval_model(ml_model): #, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, model_scores=model_scores):
def train_model(ml_model, x_train, y_train, hyper_cv, n_jobs=-1): #=y_train, x_test=x_test, y_test=y_test, model_scores=model_scores):

    print_header(ml_model, '-')

    ML_MODEL = ml_models[ml_model]

    # Record training time:
    start_t = datetime.now()

    # Train model with hyperparameter tuning:
    trained_model, training_stats = ML_MODEL.tune_train_model(x_train, y_train, hyper_cv, n_jobs=n_jobs) #, strat=False)

    #cv_tuning_stats[ml_model][n] = training_stats

    end_t = datetime.now()
    training_stats['training_t'] = end_t - start_t

    print('Training time: {}'.format(training_stats['training_t']))

    # Record tuned huperparameters:
    final_params = ML_MODEL.get_trained_params(trained_model)

    return trained_model, training_stats, final_params 

#    cv_tuning_stats[ml_model][n] = training_stats
#
#    y_pred = {'train' : trained_model.predict(x['train']).squeeze(),
#              'test'  : trained_model.predict(x['test']).squeeze()}

# Train and test in one function:
#def eval_model(trained_model, x, y_true, y_pred, model_scores, c=None, cls_labels=None):
#
#    for data_split in ['test', 'train']:
#        model_scores[data_split]['Total'] = sc.calc_model_scores(y_true[data_split], y_pred[data_split], model_scores[data_split]['Total'])
#
#    # Save predictions and scores for each class if c given (even if train/test splits not stratified):
#    if cls_labels is not None:
#    #if strat == True:
#        for cls in cls_labels:
#            y_true_cls = {'train' : np.array([y_true['train'].iloc[i] for i, c_i in enumerate(c['train']) if c_i == cls]),
#                          'test'  : np.array([y_true['test'].iloc[i]  for i, c_i in enumerate(c['test']) if c_i == cls])}
#            y_pred_cls = {'train' : np.array([y_pred['train'][i] for i, c_i in enumerate(c['train']) if c_i == cls]),
#                          'test'  : np.array([y_pred['test'][i] for i, c_i in enumerate(c['test']) if c_i == cls])}
#            if len(y_true_cls['test']) >= 2:
#                model_scores['test'][cls] = sc.calc_model_scores(y_true_cls['test'], y_pred_cls['test'], model_scores['test'][cls])
#            #print('Scores on training set:')
#            if len(y_true_cls['train']) >= 2:
#                model_scores['train'][cls] = sc.calc_model_scores(y_true_cls['train'], y_pred_cls['train'], model_scores['train'][cls])

#def train_eval_model(ml_model, x, y, hyper_cv, n_jobs=-1):
#
#    trained_model, cv_tuning_stats[ml_model][n], tuned_params[ml_model][n] = train_model(ml_model, x['train'], y['train'], hyper_cv, n_jobs)
#
#    y_pred = {'train' : trained_model.predict(x['train']).squeeze(),
#              'test'  : trained_model.predict(x['test']).squeeze()}
#
#    eval_model(trained_model, x, y, y_pred, model_scores[ml_model], c=None, cls_labels=None)
#
#    # Save individual predictions:
#    all_preds[ml_model]['train'][n][train_idx] = y_pred['train']
#    all_preds[ml_model]['test'][n][test_idx] = y_pred['test']


def eval_model(trained_model, x, y_true, y_pred, model_scores, c=None, cls_labels=None):

    print('Scores:')
    model_scores['Total'] = sc.calc_model_scores(y_true, y_pred, model_scores['Total'])

    # Save predictions and scores for each class if c given (even if train/test splits not stratified):
    if cls_labels is not None:
    #if strat == True:
        for cls in cls_labels:
            y_true_cls = np.array([y_true.iloc[i] for i, c_i in enumerate(c) if c_i == cls])
            y_pred_cls = np.array([y_pred[i] for i, c_i in enumerate(c) if c_i == cls])

            if len(y_true_cls) >= 2:
                print(cls)
                model_scores[cls] = sc.calc_model_scores(y_true_cls, y_pred_cls, model_scores[cls], print_scores=False)

    return model_scores

def train_eval_model(ml_model, x, y, hyper_cv, n_jobs=-1):

    trained_model, cv_tuning_stats[ml_model][n], tuned_params[ml_model][n] = train_model(ml_model, x['train'], y['train'], hyper_cv, n_jobs)

    for data_split in ['test', 'train']:
        y_pred = trained_model.predict(x[data_split]).squeeze()
        model_scores[ml_model][data_split] = eval_model(trained_model, x[data_split], y[data_split], y_pred, model_scores[ml_model][data_split], c=c_split[data_split], cls_labels=cls_labels)
        data_idx = {'test' : test_idx, 'train' : train_idx}
        all_preds[ml_model][data_split][n][data_idx[data_split]] = y_pred

#    y_pred = {'train' : trained_model.predict(x['train']).squeeze(),
#              'test'  : trained_model.predict(x['test']).squeeze()}
#
#    model_scores[ml_model]['train'] = eval_model(trained_model, x['train'], y['train'], y_pred['train'], model_scores[ml_model]['train'], c=c_split['train'], cls_labels=cls_labels)
#    model_scores[ml_model]['test'] = eval_model(trained_model, x['test'], y['test'], y_pred['test'], model_scores[ml_model]['test'], c=c_split['test'], cls_labels=cls_labels)
#    #eval_model(trained_model, x, y, y_pred, model_scores[ml_model], c=None, cls_labels=None)
#
#    # Save individual predictions:
#    all_preds[ml_model]['train'][n][train_idx] = y_pred['train']
#    all_preds[ml_model]['test'][n][test_idx] = y_pred['test']


################
# Main program #
################

# Work around to deal with groups
# -------------------------------

if group == True:
    # Renumber groups so that numbers start from 0 and are continuous:
    g_renum = []
    g_dups = {}
    i = 0
    for g_i in g:
        if g_i in g_dups.keys():
            j = dups[g_i]
        else:
            j = i
            g_dups[g_i] = i
            i += 1
        g_renum.append(j)
    g = np.array(g_renum)

    # Work around to deal with groups:
    g_trans = [[] for _ in set(g)]
    for i, g_i in enumerate(g):
        g_trans[g_i].append(i)
    g_trans = np.array(g_trans)

    x_red = list(range(len(set(g))))

    if strat == True:
        c_red = [c.iloc[i[0]] for i in g_trans]
    else:
        c_red = [0 for _ in range(x.shape[0])]

else:
    x_red = list(range(x.shape[0]))
    if strat == True:
        c_red = c
    else:
        c_red = [0 for _ in range(x.shape[0])]

# Random seed for train/test splitting:
split_rand_seed = 12

# No nested CV:
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_frac, random_state=split_rand_seed)

# MC CV:
if strat == True:
    train_test_cv = StratifiedShuffleSplit(mc_cv, test_size=test_frac, random_state=split_rand_seed)
    iter_splits = train_test_cv.split(x_red, c_red)
else:
    train_test_cv = ShuffleSplit(mc_cv, test_size=test_frac, random_state=split_rand_seed)
    iter_splits = train_test_cv.split(x_red)

for n, [train_idx, test_idx] in enumerate(iter_splits):

    if group == True:
        # Add data from groups back into train and test sets:
        train_idx = [i for j in g_trans[train_idx] for i in j]
        test_idx = [i for j in g_trans[test_idx] for i in j]

    # Get train and test sets:
    x_split = {'train' : x.iloc[train_idx],
               'test' : x.iloc[test_idx]}
    y_split = {'train' : y.iloc[train_idx],
               'test' : y.iloc[test_idx]}
    if c_col is not None or 'c' in col_order:
        c_split = {'train' : c.iloc[train_idx],
                   'test' : c.iloc[test_idx]}
    else:
        c_split = {'train' : None,
                   'test' : None}

    # --------------------------#
    # Train and validate models #
    # --------------------------#

    if prev_feat_select is None:
        #sys.exit('ERROR: Cannot do RFE if using previously selected features.')

        # Remove constant descriptors:
        x_split['train'], x_split['test'], n_const_desc = rm_const_desc(x_split['train'], x_split['test'])

        # Don't scale here, do this within model:
        ## Scale x data:
        #x_split['train'], x_split['test'] = scale_x(x_split['train'], x_split['test'])

        # Remove correlated variables:
        n_corr_var = np.nan
        if corr_cut is not None:
            x_split['train'], x_split['test'], n_corr_var = rm_corr_var(corr_cut, x_split['train'], x_split['test'])

        # Train models before RFE:
        for ml_model in np.array(ml_model_order)[~rfe_mask]:

            train_eval_model(ml_model, x_split, y_split, hyper_cv, n_jobs=n_jobs)
   
        if True in rfe_mask:
        
            # Do RFE:
            x_split['train'], x_split['test'] = rfe_select(x_split['train'], y_split['train'], x_split['test'], n_jobs=n_jobs)

            # Record RFE results:
            rfe_results['features'][n] = x_split['train'].columns.to_list()
            rfe_results['test_idx'][n] = test_idx

        for ml_model in np.array(ml_model_order)[rfe_mask]:

            train_eval_model(ml_model, x_split, y_split, hyper_cv, n_jobs=n_jobs)

    else:
        # Load previous feature selection from file:
        if not np.array_equal(test_idx, rfe_results['test_idx'][n]):
            sys.exit('ERROR: Previous train/test split does not match current split.')
        else:
            #n_const_desc = rfe_results['n_const_desc'][n]
            #n_corr_var = rfe_results['n_corr_var'][n]
            #select_feat = rfe_results['features'][n]

            # Select features:
            x_split['train'] = x_split['train'].loc[:,select_feat]
            x_split['test'] = x_split['test'].loc[:,select_feat]

        for ml_model in ml_model_order:
            train_eval_model(ml_model, x_split, y_split, hyper_cv, n_jobs=-1)

    # ------------ #
    # Save results #
    # ------------ #

    # Save individual predictions during each MC CV cycle:
    # ----------------------------------------------------

    # Column order is: test set scores; final features; training set scores; tuning stats

    # Write hyper parameters and predictions from each MC CV cycle:
    for ml_model in ml_model_order:
        out = mc_cv_stats_outfile[ml_model]

        out.write("{}".format(n))

        # Performance metrics:
        for metric in metric_ls:
            out.write("{}{:.5e}".format(dlim, model_scores[ml_model]['test']['Total'][metric][n]))

        # Probably better to put these in a separate file:
        # Final features (from RFE and removing correlated variables):
        out.write("{}{}".format(dlim, n_const_desc))
        out.write("{}{}".format(dlim, n_corr_var))
        if rfe_mask[ml_model_order.index(ml_model)] == True:
            out.write("{}{}{}{}".format(dlim, x_split['train'].shape[1],
                                        dlim, x_split['train'].columns
                                       ))

        # Performance metrics on training set:
        for metric in metric_ls:
            out.write("{}{:.5e}".format(dlim, model_scores[ml_model]['train']['Total'][metric][n]))

        out.write("{}{}".format(dlim, tuned_params[ml_model][n]))
        out.write("{}{}".format(dlim, cv_tuning_stats[ml_model][n]))
        out.write("\n")
        # Make sure file is written at the end of each CV cycle:
        out.flush()

        # Write test and training set predictions:
        for data_split in ['test', 'train']:
            out = preds_outfile[ml_model][data_split]
            out.write(dlim.join([str(i) if i is not np.nan else '' for i in all_preds[ml_model][data_split][n]]))
            out.write("\n")
            out.flush()

# Close files containing all predictions after final MC CV cycle:
for ml_model in ml_model_order:
    mc_cv_stats_outfile[ml_model].close()
    preds_outfile[ml_model]['test'].close()
    preds_outfile[ml_model]['train'].close()

# if save_rfe == True:
if prev_feat_select is None:
    pk.dump(rfe_results, open(rfe_results_file, 'wb'))


# Save overall scores:
# --------------------

for ml_model in ml_model_order:

    #out.write(dlim.join(df_mi.columns))
    #out.write(dlim.join([str(df_mi[col]) for col in df_mi.columns]))

#    av_stats_outfile[ml_model] = open(av_stats_outfile[ml_model], 'w')
    out = open(av_stats_outfile[ml_model], 'w')

    # Write header:
    header1 = ["Property", "Model", "Strat_class", "Datapoints", "y_stddev"]
    out.write(dlim.join(header1))

    for metric in metric_ls:
        out.write("{}{}_{}{}{}_{}_mean{}{}_{}_stddev{}{}_{}_stderr".format(dlim, metric, 'test', 
                                                                           dlim, metric, 'test', 
                                                                           dlim, metric, 'test', 
                                                                           dlim, metric, 'test'))

    header2 = ["Dataset", "Descriptors", "Datafile", "sklearn_version", "Dataset_notes", 
               "Av_(min_max)_predictions_per_mol", "MC_CV", 'hyper_cv', 'Strat_split', 'corr_cut']
    out.write(dlim)
    out.write(dlim.join(header2))

    for metric in metric_ls:
        out.write("{}{}_{}{}{}_{}_mean{}{}_{}_stddev{}{}_{}_stderr".format(dlim, metric, 'training', 
                                                                           dlim, metric, 'training', 
                                                                           dlim, metric, 'training', 
                                                                           dlim, metric, 'training'))

    # Write model name, number of datapoints and number of predictions per molecule:
    n_preds_per_mol = [sum(~np.isnan(all_preds[ml_model]['test'][:,i])) for i in range(len(x))]

    out.write("\n{}{}{}".format(prpty,
                                dlim, ml_model))

    out.write(dlim+'full_dataset')
    out.write("{}x{}_y{}".format(dlim, x.shape, y.shape))
    if group == True:
        out.write("_g({})".format(len(set(g))))
    if c_col is not None or 'c' in col_order:
        out.write("_c({})".format(len(set(c))))

    out.write('{}{}'.format(dlim, np.var(y)**0.5))

    # Calculate metrics based on all predictions from all MC CV rounds:

    # Set up dict to store scores calculated from all predictions:
    all_preds_scores = {'test' : {}, 'train' : {}}
    y_pred = {}
    y_true = {}
    for data_split in ['test', 'train']:

        y_pred[data_split] = np.array([j for i in range(y.shape[0]) for j in all_preds[ml_model][data_split][~np.isnan(all_preds[ml_model][data_split][:,i]),i]])

        y_true[data_split] = np.array([y.iloc[i] for i in range(y.shape[0]) for _ in all_preds[ml_model][data_split][~np.isnan(all_preds[ml_model][data_split][:,i]),i]])

#              'train' : np.array([j for i in range(y.shape[0]) for j in all_preds[ml_model]['train'][~np.isnan(all_preds[ml_model]['train'][:,i]),i]])}
#
#    y_true = {'test' : np.array([y.iloc[i] for i in range(y.shape[0]) for _ in all_preds[ml_model]['test'][~np.isnan(all_preds[ml_model]['test'][:,i]),i]]),
#              'train' : np.array([y.iloc[i] for i in range(y.shape[0]) for _ in all_preds[ml_model]['train'][~np.isnan(all_preds[ml_model]['train'][:,i]),i]])}

#    y_pred = {'test' : np.array([j for i in range(y.shape[0]) for j in all_preds[ml_model]['test'][~np.isnan(all_preds[ml_model]['test'][:,i]),i]]),
#              'train' : np.array([j for i in range(y.shape[0]) for j in all_preds[ml_model]['train'][~np.isnan(all_preds[ml_model]['train'][:,i]),i]])}
#
#    y_true = {'test' : np.array([y.iloc[i] for i in range(y.shape[0]) for _ in all_preds[ml_model]['test'][~np.isnan(all_preds[ml_model]['test'][:,i]),i]]),
#              'train' : np.array([y.iloc[i] for i in range(y.shape[0]) for _ in all_preds[ml_model]['train'][~np.isnan(all_preds[ml_model]['train'][:,i]),i]])}

        # Calculate scores:
        for metric in metric_ls:
            all_preds_scores[data_split][metric] = []
        all_preds_scores[data_split] = sc.calc_model_scores(y_true[data_split], y_pred[data_split], all_preds_scores[data_split])
        
        # Calculate and write total and mean scores and standard deviation and standard errors:
        for metric in metric_ls:
            mean = np.mean(model_scores[ml_model][data_split]['Total'][metric])
            stddev = calc_stddev(model_scores[ml_model][data_split]['Total'][metric])
            stderr = calc_stderr(model_scores[ml_model][data_split]['Total'][metric])

        # Only output test scores at this point:
            if data_split == 'test': #, 'train']:
                out.write("{}{:.5e}{}{:.5e}{}{:.5e}{}{:.5e}".format(dlim, all_preds_scores[data_split][metric][0], dlim, mean, dlim, stddev, dlim, stderr))

    out.write("{}{}{}{}{}{}{}{}{}{}".format(dlim, dataset,
                                            dlim, desc_set,
                                            dlim, data_file,
                                            dlim, sklearn_ver,
                                            dlim, dataset_notes,
                                         ))

    out.write("{}{:.5f}_({}_{})".format(dlim, np.mean(n_preds_per_mol), min(n_preds_per_mol), max(n_preds_per_mol)))

    out.write("{}{}{}{}{}{}{}{}".format(dlim, mc_cv, dlim, hyper_cv, dlim, strat, dlim, corr_cut))


    # Calculate and write total and mean scores and standard deviation and standard errors:
    for metric in metric_ls:
        data_split = 'train'
        out.write("{}{:.5e}{}{:.5e}{}{:.5e}{}{:.5e}".format(dlim, all_preds_scores[data_split][metric][0], dlim, mean, dlim, stddev, dlim, stderr))

    # Save stratified scores:
    # -----------------------

    if c_col is not None or 'c' in col_order:
        # Calculate and write mean scores and standard errors:
        for cls in cls_labels:

            n_preds_per_cls = [sum(~np.isnan(all_preds[ml_model]['test'][:,i])) for i in range(len(x))]

            out.write("\n{}{}{}".format(prpty,
                                        dlim, ml_model))

            out.write("{}{}".format(dlim, cls))
            out.write("{}x{}_y{}".format(dlim, x.loc[c == cls].shape, y.loc[c == cls].shape))
            if group == True:
                out.write("_g({})".format(len(set(g.loc[c == cls]))))

            out.write('{}{}'.format(dlim, np.var(y.loc[c == cls])**0.5))


            # Get all predictions and actual values for each stratified class:
            # Set up dict to store scores calculated from all predictions for given stratified class:
            all_preds_scores = {'test' : {}, 'train' : {}}
            y_pred_cls = {'test' : {}, 'train' : {}}
            y_true_cls = {'test' : {}, 'train' : {}}

            for data_split in ['test', 'train']:

                y_pred_cls[data_split] = np.array([j for i, c_i in enumerate(c) for j in all_preds[ml_model][data_split][~np.isnan(all_preds[ml_model][data_split][:,i]),i] if c_i == cls])
                y_true_cls[data_split] = np.array([y.iloc[i] for i, c_i in enumerate(c) for _ in all_preds[ml_model][data_split][~np.isnan(all_preds[ml_model][data_split][:,i]),i] if c_i == cls])

                # Calculate scores:
                for metric in metric_ls:
                    all_preds_scores[data_split][metric] = []
                all_preds_scores[data_split] = sc.calc_model_scores(y_true_cls[data_split], y_pred_cls[data_split], all_preds_scores[data_split])
                # Write scores:
                for metric in metric_ls:
                # Write scores based on all predcitions:
                # Calculate and write mean scores and standard errors, averaged over MC CV cycles:
                    mean = np.mean(model_scores[ml_model][data_split][cls][metric])
                    stddev = calc_stddev(model_scores[ml_model][data_split][cls][metric])
                    stderr = calc_stderr(model_scores[ml_model][data_split][cls][metric])

                    if data_split == 'test':
                        out.write("{}{:.5e}{}{:.5e}{}{:.5e}{}{:.5e}".format(dlim, all_preds_scores[data_split][metric][0], dlim, mean, dlim, stddev, dlim, stddev, dlim, stderr))

            out.write("{}{}{}{}{}{}{}{}{}{}".format(dlim, dataset,
                                                    dlim, desc_set,
                                                    dlim, data_file,
                                                    dlim, sklearn_ver,
                                                    dlim, dataset_notes))
            
            out.write("{}{:.5f}_({}_{}){}{}".format(dlim, np.mean(n_preds_per_cls), min(n_preds_per_cls), max(n_preds_per_cls), dlim, mc_cv, ))
            out.write("{}{}{}{}{}{}{}{}".format(dlim, mc_cv, dlim, hyper_cv, dlim, strat, dlim, corr_cut))

            # Calculate scores:
            for metric in metric_ls:
                data_split = 'train' #, 'train']:
                out.write("{}{:.5e}{}{:.5e}{}{:.5e}{}{:.5e}".format(dlim, all_preds_scores[data_split][metric][0], dlim, mean, dlim, stddev, dlim, stderr))
#    out.close()


#def write_results(outfile, y=y, model_scores=model_scores):
#    out = open(outfile, 'w')
#
#    # Write header:
#    header1 = ["Property", "Model", "Datapoints", "y_stddev"]
#    out.write(dlim.join(header1))
#
#    for metric in metric_ls:
#        out.write("{}{}_{}".format(dlim, metric, 'testext'))
#
#    header2 = ["Dataset", "Descriptors", "Datafile", "sklearn_version", "Dataset_notes"]
#    out.write(dlim)
#    out.write(dlim.join(header2))
#
#    out.write("\n{}{}{}".format(prpty,
#                                dlim, ml_model))
#
#    out.write("{}x{}_y{}".format(dlim, x_ext.shape, y_ext.shape))
#    if group == True:
#        out.write("_g({})".format(len(set(g))))
#    if 'c' in col_order:
#        out.write("_c({})".format(len(set(c))))
#
#    out.write('{}{}'.format(dlim, np.var(y)**0.5))
#
#    # Calculate and write total and mean scores and standard deviation and standard errors:
#    for metric in metric_ls:
#        out.write("{}{:.5e}".format(dlim, model_scores[metric][0]))
#
#    out.write("{}{}{}{}{}{}{}{}{}{}".format(dlim, dataset,
#                                            dlim, desc_set,
#                                            dlim, data_file,
#                                            dlim, sklearn_ver,
#                                            dlim, dataset_notes,
#                                           ))
#
#    out.close()

def write_results(filename):

    out = open(filename, 'w')

    model_details = [("Model", ml_model),
                     ("Test_fraction", test_frac),
                     ("MC_CV", mc_cv),
                     ("Hyper_CV", hyper_cv),
                    # ("Predictions_file", pred_file),
                    ]
    dataset_details = [("Property", prpty),
                       ("Training_dataset", data_file),
                       ("Test_dataset", ''),
                       ("Descriptors", desc),
                       ("pH", ph),
                       ("pH_model", phmodel),
                      ]
    feat_select_details = []
#    if prev_feat_select is not None:
#        use_rfe = rfe_mask[ml_model_order.index(ml_model)]
#        feat_select_details = [("Correlation_cutoff", corr_cut),
#                               ("N_correlated_descs", n_corr_var),
#                               ("RFE", use_rfe),
#                              ]
#        if use_rfe:
#            feat_select_details += [("RFE_min", rfe_min),
#                                    ("RFE_max", rfe_max),
#                                    ("RFE_cv", rfe_cv),
#                                    ("RFE_strat", rfe_strat),
#                                   ]
    scoring_details = [("y_stddev", np.var(y)**0.5),
                      ]
    training_details = [("N_jobs", n_jobs),
                       ]

    for title, val in model_details + dataset_details + feat_select_details + scoring_details:
        out.write('{}{}{}\n'.format(title, dlim, val))

    for metric in metric_ls:
#        for data_split in ['test', 'train']:
#            mean = np.mean(model_scores[ml_model][data_split]['Total'][metric])
#            stddev = calc_stddev(model_scores[ml_model][data_split]['Total'][metric])
#            stderr = calc_stderr(model_scores[ml_model][data_split]['Total'][metric])
#
#            out.write('{}{}{}'.format(metric+'_'+data_split, dlim, mean))

        mean = model_scores[ml_model][metric]
        out.write('{}{}{}\n'.format(metric+'_'+data_split, dlim, mean))

    #write scores

    # Write individual predictions (horizontal):
    #out = open('', 'w')
    #out.write(dlim.join([str(i) for i in y_ext.index]))
    #out.write("\n")
    #out.write(dlim.join([str(i) for i in y_pred]))
    #out.close()

#    out = open('', 'w')
#    out.write('{}{}{}'.format('', dlim, ''))
#    #for out.write('{}{}{}'.format('', dlim, ''))
#
#    out.write(dlim.join([str(i) for i in y_ext.index]))
#    out.write("\n")
#    out.write(dlim.join([str(i) for i in y_pred]))
#    out.close()

#def write_results(header, scores={}):
#    for col in header[0]:
#        if col in header[1]:
#            write(dlim.join(col+'_'+sub for sub in header[1][col]))
#        else:
#            write(col)
#    write('\n')
#    if scores is None:
#        return
#    for col in header[0]:
#        if col in header[1]:
#            write()
#        else:
#            write(scores[col])


#write_results(out, header1 + ['SCORES'] + header2, model_scores[ml_model])

def write_results(out, header, di, scores, dlim=';', write_head=False):
#    out = open(av_stats_outfile[ml_model], 'w')
#    if write_head=True:
#        for col_i, col in enumerate(header):
#            if col_i != 0:
#                out.write(dlim)
#             out.write(col)
    out.write('\n')
    for col_i, col in enumerate(header):
        if col_i != 0:
            out.write(dlim)
        if col not in ['SCORES']:
            out.write(di.get(col))
        elif col == 'SCORES':
            for metric in metric_ls:
                out.write(dlim.join([str(scores[metric]), '', '', '']))
#                write(mean)



if '-refit' in sys.argv:

    cv_tuning_stats = {}
    trained_models = {}
    tuned_params = {}
    model_scores = {}

    x, n_rm_descs = rm_const_desc(x)

    n_corr_var = np.nan
    if corr_cut is not None:
        x, n_corr_var = rm_corr_var(corr_cut, x)

    for ml_model in np.array(ml_model_order)[~rfe_mask]:
        cv_tuning_stats[ml_model] = {}
        tuned_params[ml_model] = {}
        model_scores[ml_model] = {}
        for metric in metric_ls:
            model_scores[ml_model][metric] = []

        trained_models[ml_model], cv_tuning_stats[ml_model], tuned_params[ml_model] = train_model(ml_model, x, y, hyper_cv, n_jobs)
#tune_train_model(x, y)
        pk.dump(trained_models[ml_model], open(ml_model+'_refit.pk', 'wb'))

        if '-ext_test' in sys.argv:
            x_ext = {}
            y_ext = {}
            for ext_name, data_file in ext_ls:

                if calc_descs:
                    df = pd.read_csv(data_file)

                    df_desc = calc_desc_df(df[smi_col], tauto, ph, desc_ls=x.columns.to_list())

                    x_ext[ext_name] = df_desc.drop(['Processed_SMILES', 'Warnings'], axis=1)

                    df.set_index(smi_col, inplace=True, verify_integrity=True)
                    y_ext[ext_name] = df[y_col]

                else:
                    ext_data = pk.load(open(data_file, "rb"))
                    x_ext[ext_name] = ext_data[col_order.index('x')]
                    y_ext[ext_name] = ext_data[col_order.index('y')]

            for ext_name in ext_ls[:,0]:
                y_pred = trained_models[ml_model].predict(x_ext[ext_name])
                model_scores[ml_model] = sc.calc_model_scores(y_ext[ext_name], y_pred, model_scores[ml_model])
                write_results(out, header1 + ['SCORES'] + header2, di, model_scores[ml_model])
                #write_results(data_file+'_'+ml_model)

    if True in rfe_mask:
        x = rfe_select(x, y)
        if '-ext_test' in sys.argv:
            for ext_name in ext_ls[:,0]:
                x_ext[ext_name] = x_ext[ext_name][[col for col in x.columns]]

    for ml_model in np.array(ml_model_order)[rfe_mask]:
        cv_tuning_stats[ml_model] = {}
        tuned_params[ml_model] = {}
        model_scores[ml_model] = {}
        for metric in metric_ls:
            model_scores[ml_model][metric] = []

        trained_models[ml_model], cv_tuning_stats[ml_model], tuned_params[ml_model] = train_model(ml_model, x, y, hyper_cv, n_jobs)
        #tune_train_model(x, y)
        pk.dump(trained_models[ml_model], open('_refit.pk', 'wb'))

        if '-ext_test' in sys.argv:
            for ext_name in ext_ls[:,0]:
                y_pred = trained_models[ml_model].predict(x_ext[ext_name])
                model_scores[ml_model] = sc.calc_model_scores(y_ext[ext_name], y_pred, model_scores[ml_model])
                write_results(out, header1 + ['SCORES'] + header2, di, model_scores[ml_model])
                #write_results(data_file+'_'+ml_model)

#    if '-ext_test' in sys.argv:
#        for data_file in ext_testset_files:
#
#            if calc_descs:
#                df = pd.read_csv(data_file)
#
#                df_desc = calc_desc_df(df[smi_col], tauto, ph, desc_ls=x.columns.to_list())
#
#                x_ext = df_desc.drop(['Processed_SMILES', 'Warnings'], axis=1)
#
#                df.set_index(smi_col, inplace=True, verify_integrity=True)
#                y_ext = df[y_col]
#
#            else:
#                ext_data = pk.load(open(data_file, "rb"))
#                x_ext = ext_data[col_order.index('x')]
#                y_ext = ext_data[col_order.index('y')]
#
#            for ml_model in np.array(ml_model_order)[~rfe_mask]:
#
#                # Do feature selection:
#                #x_ext = x_ext[[col for col in x.columns]]
#                x_ext = x_ext[[col for col in x_no_rfe]]
#
#                y_pred = trained_models[ml_model].predict(x_ext)
#
#                model_scores[ml_model] = sc.calc_model_scores(y_ext, y_pred, model_scores[ml_model])
#
#                write_results(data_file+'_'+ml_model) #, y_ext, model_scores[ml_model])
#
#                # Write predictions:
#                out = open(data_file+'_'+ml_model+'_pred', 'w')
#                out.write(dlim.join([str(i) for i in y_ext.index]))
#                out.write("\n")
#                out.write(dlim.join([str(i) for i in y_pred]))
#                out.write("\n")
#                out.close()
#
#            if True in rfe_mask:
#                x_ext = rfe_select(x_ext, y_ext)
#
#            for ml_model in np.array(ml_model_order)[~rfe_mask]:
#
#                # Do feature selection:
#                #x_ext = x_ext[[col for col in x.columns]]
#
#                y_pred = trained_models[ml_model].predict(x_ext)
#
#                model_scores[ml_model] = sc.calc_model_scores(y_ext, y_pred, model_scores[ml_model])
#
#                write_results(data_file+'_'+ml_model) #, y_ext, model_scores[ml_model])
#
#                # Write predictions:
#                out = open(data_file+'_'+ml_model+'_pred', 'w')
#                out.write(dlim.join([str(i) for i in y_ext.index]))
#                out.write("\n")
#                out.write(dlim.join([str(i) for i in y_pred]))
#                out.write("\n")
#                out.close()

