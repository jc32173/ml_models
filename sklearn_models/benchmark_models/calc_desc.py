# Writen October 2020

from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors

import numpy as np
import pandas as pd
import re

from smi_funcs import canonicalise_tautomer, correct_smiles, \
    adjust_for_ph, calc_rdkit_descs


# Process SMILES before calculating descriptors 
# (canonicalise tautomer, correct smiles, adjust for ph)
def process_smiles(smi, tauto=False, ph=None, phmodel=None):
    warnings = ''

#    if tauto == 'OpenEye':
#        mol = 

    if tauto:
        # Convert to canonical tautomer using rdkit:
        smi, warning = canonicalise_tautomer(smi)
        if warning != '':
            warnings += warning

    # Correct known errors in SMILES from canonicalizeTautomers in rdkit:
    smi, warning = correct_smiles(smi, [[r"\[PH\]\(=O\)\(=O\)O", 
                                         r"P(=O)(O)O"]])
    if warning != '':
        warnings += warning

    # Change tetrazole tautomer by moving H to first N (i.e. c1[nH]nnn1)
    # so that it can be correctly deprotonated in obabel if pH < pKa:
    smi, warning = correct_smiles(smi, [[r"c(\d+)nn\[nH\]n(\d+)", 
                                         r"c\1[nH]nnn\2"]])
    if warning != '':
        warnings += warning

    if ph: # is not None:
        # Convert SMILES to given pH:
        smi, warning = adjust_for_ph(smi, ph=ph, phmodel=phmodel)
        if warning != '':
            warnings += warning

    # Make SMILES canonical?

    return smi, warnings

# Calculate molecular descriptors used in logD model:
def calc_desc(smi, desc_ls, tauto, ph, phmodel):
    warnings = ''

    smi, warning = process_smiles(smi, tauto, ph, phmodel)
    if warning != '':
        warnings += warning

    # Calculate rdkit descriptors:
    descs, warning = calc_rdkit_descs(smi, desc_ls)
    if warning != '':
        warnings += warning
    return [smi, warnings] + list(descs)

def calc_desc_df(smi_ls,
                 id_ls=True, 
                 tauto=True, 
                 ph=None, 
                 phmodel=None, 
                 desc_ls=None, 
                 rm_na=True):
    """
    Run calc_desc on a DataFrame
    """
    if desc_ls is None:
        desc_ls = [x[0] for x in Chem.Descriptors._descList]

    if isinstance(smi_ls, list):
        df_desc = pd.DataFrame(data=smi_ls, columns=['SMILES'])
        smi_col = 'SMILES'

    elif isinstance(smi_ls, pd.core.series.Series):
        df_desc = pd.DataFrame(smi_ls)
        smi_col = smi_ls.name

    # Check SMILES are valid?
    df_desc['Mol'] = df_desc.apply(lambda row: Chem.MolFromSmiles(row[smi_col]), axis=1)
    df_desc = df_desc.loc[~df_desc['Mol'].isna()]
    df_desc.drop(columns=['Mol'], inplace=True)

    # Preprocess SMILES and calculate descriptors:
    df_desc[['Processed_SMILES', 'Warnings'] + desc_ls] = df_desc.apply(lambda row: 
        calc_desc(row[smi_col], desc_ls, tauto, ph, phmodel), axis=1, result_type='expand')
    if id_ls:
        df_desc.set_index(smi_col, inplace=True, verify_integrity=True)

    # Check no NaN in descriptors and remove if found:
    if rm_na:
        df_desc_only = df_desc[[col for col in df_desc 
                                if col not in ['Processed_SMILES', 'Warnings']]]
        if df_desc_only.isnull().values.any():
            print('WARNING: Initial set of descriptors contained NaN values:')
            print('Descriptor\tNumber of molecules with NaN values')
            for d_nan in df_desc_only.columns[df_desc_only.isnull().any()]:
                print(d_nan+'\t'+sum(df_desc_only[d_nan].isna()))
            print('These will be removed')
            df_desc = df_desc.drop(df_desc_only.columns[df_desc_only.isnull().any()], axis=1, inplace=True)

    return df_desc


def calc_mordred_df(smi_ls):
    calc = mordred.Calculator(mordred.descriptors, ignore_3D=True)
    df_desc = calc.pandas(df_train['Mol'])
    return df_desc
