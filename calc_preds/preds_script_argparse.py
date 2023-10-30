#! /bin/env python

# November 2022
# Script to run calc_preds_desc.py and get predictions, descriptors and
# substructure matches for large dataset of InChIs.


# Path to PP_ML_models directory, must end in "/":
#PP_ML_models_path = "/users/xpb20111/programs/ml_model_code/PP_ML_models/"
PP_ML_models_path = '/users/xpb20111/programs/ml_model_code/'

import sys, os
import numpy as np
import pandas as pd
# Silence SettingWithCopyWarning from pandas (see: https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas):
pd.options.mode.chained_assignment = None
import argparse
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

# Should be able to get RDContribDir from: 
try:
    from rdkit.Chem import RDConfig
    RDContribDir = RDConfig.RDContribDir
    import sascorer
except ModuleNotFoundError:
    # but isn't pointing to the right directory in deepchem environment so hard-code instead:
    try:
        RDContribDir = '/users/xpb20111/.conda/envs/deepchem/share/RDKit/Contrib/'
        sys.path.append(os.path.join(RDContribDir, 'SA_Score'))
        import sascorer
    except ModuleNotFoundError:
        sascorer = False

sys.path.insert(0, '/users/xpb20111/programs/')
from cheminfo_utils.rdkit_extra_desc import GetFusedRings
sys.path.insert(0, '/users/xpb20111/programs/ml_model_code/calc_predictions_on_dataset')
from calc_preds_descs import process_df, class_sol, error_wrapper, substruct_match

#sys.path.insert(0, '/users/xpb20111/programs/ml_model_code/2023.7.3')
sys.path.insert(0, '/users/xpb20111/programs/deepchem_dev_nested_CV/deepchem_models/final_models/')
from ml_model_gcnn import Ensemble_Model_DC



# Possible descriptors to calculate (all take RDKit mol object as input):
descriptors={'molwt' : rdMolDescriptors.CalcExactMolWt, 
             'n_aromatic_rings' : rdMolDescriptors.CalcNumAromaticRings, 
             'n_heavy_atoms' : rdMolDescriptors.CalcNumHeavyAtoms, 
             'murcko_scaffold' : lambda mol: MurckoScaffoldSmiles(mol=mol, 
                                                                  includeChirality=False), 
             'max_fused_rings' : lambda mol: len(GetFusedRings(mol)[0]),
             '_order' : ['molwt', 
                         'n_aromatic_rings', 
                         'n_heavy_atoms', 
                         'murcko_scaffold',
                         'max_fused_rings'
                        ]
            }
if sascorer:
    descriptors['SAscore'] = sascorer.calculateScore
    descriptors['_order'].append('SAscore')

# Command line arguments:
parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument('-m', '--models', nargs='+', help="ML models to calculate predictions.")


#--------------#
# Run options: #
#--------------#

# Input file:
#infile=sys.argv[3] #'test_inchis.inchi.gz'
parser.add_argument('-i', '--inchis', dest='infile', help="Input file containing list of InChIs.")
index_col=False
index_prefix='a' # Label each dataset from Bruno with a letter
parser.add_argument('-d', '--delim', dest='sep', default=';', help="Delimiter for input file.")
parser.add_argument('-s', '--struct_col', dest='structure_col', default='InChI', help="")

# Read options:
#start_line=int(sys.argv[1])
#end_line=int(sys.argv[2])
# SET DEFAULTS FOR THIS?
parser.add_argument('-l', '--lines', nargs=2, type=int, default=(0, -1), help="Lines of input file to read.")
print('Running: '+' '.join(sys.argv))
# Number of lines to read and calculate predictions for in a batch:
#chunksize=20
parser.add_argument('-c', '--chunksize', default=1000, type=int, help="Number of lines to process at a time.")

# Get results for all possible tautomer or stereoisomer forms:
#enum_tauto=False
parser.add_argument('--enum_tauto', action='store_true', help="Evaluate all possible tautomers.")
#enum_iso=False
# CHECK enum_iso to enum_stereo IN ALL CODE:
parser.add_argument('--enum_stereo', dest='enum_iso', action='store_true', help="Evaluate all possible stereoisomers.")

# Apply Lilly rules:
#lilly_rules=True
parser.add_argument('--lilly_rules', action='store_true', help="Run Lilly rules.")
#drop_lilly_failures=False
parser.add_argument('--drop_lilly_failures', action='store_true', help="Drop compounds which fail Lilly rules (saves prediction time).")
parser.add_argument('--lilly_rules_script', default='Lilly_Medchem_Rules.rb', help="Location of Lilly_Medchem_Rules.rb script.")

# Data to save:
#outfile=f'all_preds_{start_line}-{end_line}.csv.gz'
parser.add_argument('-o', '--out', dest='outfile_prefix', default='all_preds_', help="Output file name (prefix).")
#invalid_inchi_file='invalid_inchis.inchi.gz'
parser.add_argument('-e', '--errors', dest='invalid_inchi_file', help="File to store invalid InChIs which cannot be read.")

# Calculate histograms:
            # Property : Bin width
#save_hists={'pIC50_pred' : 0.1, 
#            #'pIC50_uncert' : 0.1,
#            'MPO' : 0.1, 
#            'molwt' : 5, 
#            'n_heavy_atoms' : 1
#           }
#save_hists={}
hist_file_prefix='hist'
append_to_hist=True
#parser.add_argument('--hist', help="Histogram predictions.")
parser.add_argument('--hist', default=False, nargs='*', help="Pairs of values for property and bin width for histogramming, if none given default values are used.")


parser.add_argument('--desc', nargs='*', help='')



# Substructures to search for:
#df_substructs=None
#df_substructs=pd.read_csv('/users/xpb20111/Prosperity_partnership/pymolgen_output/fragments/hydantoin_substructs.csv')
#df_substructs['substruct_mol']=[Chem.MolFromSmarts(smi) for smi in df_substructs['substruct_SMARTS']]
#hist_by_substruct=True

parser.add_argument('--hist_by_substruct', default=False, help="")


# Other values to calculate:
#calc_logp_oe=True
parser.add_argument('--calc_oe_logp', action='store_true', help="Calculate OpenEye logP descriptor.")
#calc_pfi=True
parser.add_argument('--calc_pfi', action='store_true', help="Calculate PFI.")
#calc_mpo=True
parser.add_argument('--calc_mpo', action='store_true', help="Calculate MPO.")
# Need to include options for calculated and predicted logDs


#--------------------------#
# Load predictive models: -#
#--------------------------#


#THINK MORE ABOUT THIS:
# LoadModel needs to be a general class to open pk file and then initiat specific class for model:
#models = {'_order' : []}
#for model_file in args.models:
#    model = LoadModel(model_file)
#    models[model.name] = model
#    models['_order'].append(model.name)


## pIC50:
#pIC50_pred_model = Ensemble_Model_DC('/users/xpb20111/programs/ml_model_code/2022.4.1/pIC50.pk')
#print('Using pIC50 model: {}'.format(pIC50_pred_model.version))
#print(pIC50_pred_model.info)
#_ = pIC50_pred_model.predict('C')[0]
#
## logD:
#logD_pred_model = Ensemble_Model_DC('/users/xpb20111/programs/ml_model_code/2022.4.1/logD.pk')
#print('Using logD model: {}'.format(logD_pred_model.version))
#print(logD_pred_model.info)
#_ = logD_pred_model.predict('C')[0]
#
## Sol and Perm:
## NOTE: Early models surpress stderr which prevents Traceback if 
## script exists with an error.
#code_dir = '/users/xpb20111/programs/ml_model_code/2020.1.1/'
#sys.path.insert(0, code_dir)
#from perm import Perm_Model
#from sol import Sol_Model
#sol_pred_model = Sol_Model('/users/xpb20111/programs/ml_model_code/2021.2.1/sol.pk', 
#                           'WDkw212Wab30m32l3i0qJAwYgD8qYnqV5pq7oCLd4e0=')
#perm_pred_model = Perm_Model('/users/xpb20111/programs/ml_model_code/2021.2.1/perm.pk', 
#                             'AttUnNZ0RdeQujfzaaHVocRKzowe8Camg2mvlbxt2Zk=')

models = {'pIC50_pred' : lambda smis: pIC50_pred_model.predict(smis)[0],
          #('pIC50_pred_from_uncert',
          # 'pIC50_uncert') : lambda smis: np.array(pIC50_pred_model.predict_uncertainty(smis)[:-1]).T,
          #'logD_pred' : lambda smis: logD_pred_model.predict(smis)[0], 
          #'sol_pred' : lambda smis: [class_sol(sol_pred_model.predict(smi)[0]) for smi in smis], 
          #'sol_pred' : lambda smis: 
          #             [error_wrapper(smi, lambda smi: class_sol(sol_pred_model.predict(smi)[0])) for smi in smis], 
          ##'perm_pred' : lambda smis: [perm_pred_model.predict(smi)[0] for smi in smis], 
          #'perm_pred' : lambda smis: [error_wrapper(smi, lambda smi: perm_pred_model.predict(smi)[0]) for smi in smis], 
          '_order' : ['pIC50_pred', 
                      #('pIC50_pred_from_uncert',
                      # 'pIC50_uncert'),
                      #'logD_pred', 
                      #'sol_pred', 
                      #'perm_pred'
                     ]
         }


#----------------------------------#
# Run function to get predictions: #
#----------------------------------#

# Calculate predictions:
if __name__ == '__main__':

    args = parser.parse_args()
    args.outfile_prefix = args.outfile_prefix.replace('.gz', '')
    args.outfile_prefix = args.outfile_prefix.replace('.csv', '')
    if tuple(args.lines) == (0, -1):
        outfile = f'{args.outfile_prefix}.csv.gz'
    else:
        outfile = f'{args.outfile_prefix}_{args.lines[0]}-{args.lines[1]}.csv.gz'

    def to_numeric(s):
        if '.' in s:
            return float(s)
        else:
            return int(s)

    # Process command line arguments for histograms:
    if isinstance(args.hist, list):
        if len(args.hist) % 2 != 0:
            raise ValueError('Number of values given to --hist must be even (or zero)')
        # Default values:
        elif len(args.hist) == 0:
            hist=('pIC50_pred', 0.1, 
                  'MPO', 0.1, 
                  'molwt', 5, 
                  'n_heavy_atoms', 1)
        save_hists={}
        for prpty, bin_width in zip(args.hist[::2], args.hist[1::2]):
            save_hists[prpty] = to_numeric(bin_width)

        if args.hist_by_substruct:
            df_substructs=pd.read_csv(args.hist_by_substruct)
            df_substructs['substruct_mol']=[Chem.MolFromSmarts(smi) for smi in df_substructs['substruct_SMARTS']]
            hist_by_substruct=True

    # Process command line arguments for descriptors:
    selected_desc = {name : fn for name, fn in descriptors.items() 
                     if name in args.desc}
    selected_desc['_order'] = [name for name in args.desc 
                               if name in descriptors.keys()]

    # Process command line arguments for ML models:
    sys.path.insert(0, '/users/xpb20111/programs/ml_model_code/2022.4.1')
    from predictive_models.ml_model_gcnn_ens import Ensemble_Model_DC
    sys.path.insert(0, '/users/xpb20111/programs/ml_model_code/2020.1.1')
    from perm import Perm_Model
    from sol import Sol_Model
    def load_model(model_name, pk_filename):
        if model_name in ['pIC50_pred', 'logD_pred']:
            pred_model = Ensemble_Model_DC(pk_filename)
            print('Using {} model version: {} ({})'.format(model_name, pred_model.version, pred_model.info))
            _ = pred_model.predict('C')[0]
            return lambda smis: pred_model.predict(smis)[0]
        elif model_name == 'sol_pred':
            sol_pred_model = \
            Sol_Model('/users/xpb20111/programs/ml_model_code/2021.2.1/sol.pk', 'WDkw212Wab30m32l3i0qJAwYgD8qYnqV5pq7oCLd4e0=')
            return lambda smis: [error_wrapper(smi, lambda smi: class_sol(sol_pred_model.predict(smi)[0])) for smi in smis]
        elif model_name == 'perm_pred':
            perm_pred_model = \
            Perm_Model('/users/xpb20111/programs/ml_model_code/2021.2.1/perm.pk', 'AttUnNZ0RdeQujfzaaHVocRKzowe8Camg2mvlbxt2Zk=')
            return lambda smis: [error_wrapper(smi, lambda smi: perm_pred_model.predict(smi)[0]) for smi in smis]

    if len(args.models) % 2 != 0:
        raise ValueError('--models expects pairs of values: model_name pk_filename, so must be even (or zero)')
    models = {model_name : load_model(model_name, pk_filename) for 
              model_name, pk_filename in zip(args.models[::2], args.models[1::2])}
    models['_order'] = args.models[::2]

#    # pIC50:
#    pIC50_pred_model = Ensemble_Model_DC(args.models[0])
#    print('Using pIC50 model: {}'.format(pIC50_pred_model.version))
#    print(pIC50_pred_model.info)
#    _ = pIC50_pred_model.predict('C')[0]
#
#    models = {'pIC50_pred' : lambda smis: pIC50_pred_model.predict(smis)[0],
#              '_order' : ['pIC50_pred', 
#                         ]
#             }

    process_df(args.infile, 
               sep=args.sep, 
               structure_col=args.structure_col, 
               start_line=args.lines[0], 
               end_line=args.lines[1], 
               index_col=index_col, 
               index_prefix=index_prefix, 
               chunksize=args.chunksize, 
               invalid_inchi_file=args.invalid_inchi_file, 
               save_hists=save_hists, 
               enum_tauto=args.enum_tauto, 
               enum_iso=args.enum_iso, 
               lilly_rules=args.lilly_rules,
               lilly_rules_script=args.lilly_rules_script,
               drop_lilly_failures=args.drop_lilly_failures,
               models=models, 
               descriptors=descriptors, 
               substructs=df_substructs, 
               hist_by_substruct=hist_by_substruct, 
               calc_logp_oe=args.calc_oe_logp, 
               calc_pfi=args.calc_pfi, 
               calc_mpo=args.calc_mpo, 
               outfile=outfile, 
               hist_file_prefix=hist_file_prefix, 
               append_to_hist=append_to_hist
              )
