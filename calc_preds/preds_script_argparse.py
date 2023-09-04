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
import argparse
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
# Should be able to get RDContribDir from: 
# from rdkit.Chem import RDConfig
# RDContribDir = RDConfig.RDContribDir
# but isn't pointing to the right directory in deepchem environment so hard-code instead:
RDContribDir = '/users/xpb20111/.conda/envs/deepchem/share/RDKit/Contrib/'
sys.path.append(os.path.join(RDContribDir, 'SA_Score'))
import sascorer
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

sys.path.insert(0, '/users/xpb20111/programs/')
from cheminfo_utils.rdkit_extra_desc import GetFusedRings
sys.path.insert(0, '/users/xpb20111/programs/ml_model_code/calc_predictions_on_dataset')
from calc_preds_descs import process_df, class_sol, error_wrapper, substruct_match

sys.path.insert(0, '/users/xpb20111/programs/ml_model_code/2023.7.3')
from predictive_models.ml_model_gcnn import Ensemble_Model_DC



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
save_hists={}
hist_file_prefix='hist'
append_to_hist=True
parser.add_argument('--hist', help="Histogram predictions.")

# Descriptors to calculate (take RDKit mol object as input):
descriptors={'molwt' : rdMolDescriptors.CalcExactMolWt, 
             'n_aromatic_rings' : rdMolDescriptors.CalcNumAromaticRings, 
             'n_heavy_atoms' : rdMolDescriptors.CalcNumHeavyAtoms, 
             'murcko_scaffold' : lambda mol: MurckoScaffoldSmiles(mol=mol, 
                                                                  includeChirality=False), 
             'SAscore' : sascorer.calculateScore,
             'max_fused_rings' : lambda mol: len(GetFusedRings(mol)[0]),
             '_order' : ['molwt', 
                         'n_aromatic_rings', 
                         'n_heavy_atoms', 
                         'murcko_scaffold',
                         'SAscore',
                         'max_fused_rings'
                        ]
            }

# Substructures to search for:
#df_substructs=None
df_substructs=pd.read_csv('/users/xpb20111/Prosperity_partnership/pymolgen_output/fragments/hydantoin_substructs.csv')
df_substructs['substruct_mol']=[Chem.MolFromSmarts(smi) for smi in df_substructs['substruct_SMARTS']]
hist_by_substruct=True

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
    if args.lines == (0, -1):
        outfile = f'{args.outfile_prefix}.csv.gz'
    else:
        outfile = f'{args.outfile_prefix}_{args.lines[0]}-{args.lines[1]}.csv.gz'

    # pIC50:
    pIC50_pred_model = Ensemble_Model_DC(args.models[0])
    print('Using pIC50 model: {}'.format(pIC50_pred_model.version))
    print(pIC50_pred_model.info)
    _ = pIC50_pred_model.predict('C')[0]

    models = {'pIC50_pred' : lambda smis: pIC50_pred_model.predict(smis)[0],
              '_order' : ['pIC50_pred', 
                         ]
             }

    process_df(args.infile, 
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
