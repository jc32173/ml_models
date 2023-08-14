# Apply all analysis functions:

import math
import sys, os
from rdkit import Chem
import pandas as pd
import re

sys.path.insert(0, '/users/xpb20111/programs/deepchem_dev_nested_CV')

from cheminfo_utils.rdkit_funcs import \
    GetMol, canonicalise_smiles, canonicalise_tautomer, inchi_check, \
    GetPossibleTautomers, GetPossibleStereoisomers, \
    GetPossibleTautomersStereoisomers, is_ionisable, GetEnantiomer


def find_possible_duplicates(sr_smi): #, id_col):
    """
    Read list of lists of possible SMILES and identify any
    matches, due to tautomers or possible stereoisomers.

    sr_smi: Series of lists: [ID, [smiles_1, smiles_2, ...]]
    """

    dup_idxs = [[] for _ in range(len(sr_smi))] #[[]]*len(sr_smi)
    idx = 0
    #for i_1, smi_ls_1, in enumerate(ls_col):
    for i_1, [idx_1, smi_ls_1] in enumerate(sr_smi.iteritems()):
        for j, [idx_2, smi_ls_2] in enumerate(sr_smi.iloc[i_1+1:].iteritems()):
            i_2 = i_1 + 1 + j
            #if i_1 == i_2:
            #    continue
            if len(set(smi_ls_1) & set(smi_ls_2)) > 0:
                dup_idxs[i_1].append(idx_2)
                dup_idxs[i_2].append(idx_1)
    return dup_idxs


def find_enantiomers(df_smi_enant): #, generate_enantiomers=False):
    """
    Find enantiomers within a list of SMILES.

    df_smi_enant: DataFrame with SMILES and Enantiomer columns

    To do: Generate enantiomers if not given as input.
    """

    enant_idxs = [None for _ in range(len(df_smi_enant))]
    idx = 0
    for i_1, [idx_1, [smi_1, enant_1]] in enumerate(df_smi_enant.iterrows()):
        for j, [idx_2, [smi_2, enant_2]] in \
            enumerate(df_smi_enant.iloc[i_1+1:].iterrows()):
            i_2 = i_1 + 1 + j
            if enant_1 == smi_2:
                # Double check the reverse:
                if smi_1 != enant_2:
                    raise ValueError('')
                enant_idxs[i_1] = idx_2
                enant_idxs[i_2] = idx_1
    return enant_idxs


def find_stereoisomers(sr_smi):
    """
    Find stereoisomers within a list of SMILES (non-identical SMILES which 
    share the same canonical 2D SMILES).
    """
    
    idx_order = sr_smi.index
    idx_name = sr_smi.index.name
    if idx_name is None:
        idx_name = 'index'

    df_isomer_grps = sr_smi.loc[sr_smi.duplicated(keep=False)]\
                           .reset_index(drop=False)\
                           .groupby('2D_SMILES')\
                           .agg(lambda x: list(x))

    isomers = []
    for idx_ls in df_isomer_grps[idx_name]:
        for idx in idx_ls:
            # Remove current ID from list of stereoisomers:
            isomers.append([idx, sorted(list(set(idx_ls) - {idx}))])

    df_isomers = pd.DataFrame(data=[[[]]]*len(idx_order),
                              columns=['Stereoisomers'],
                              index=idx_order)

    df_isomers_only = \
    pd.DataFrame(isomers, columns=[idx_name, 'Stereoisomers'])\
      .set_index(idx_name, verify_integrity=True)

    df_isomers.loc[df_isomers_only.index, 'Stereoisomers'] = \
        df_isomers_only['Stereoisomers']

    return df_isomers.loc[idx_order]

# def silence_print(fn):
#     def redirect_stdout(*args, **kwags):
#         old_stdout = sys.stdout # backup current stdout
#         sys.stdout = open(os.devnull, "w")
#         fn(*args, **kwags)
#         sys.stdout = old_stdout # reset old stdout

# def silence_print(fn, *args, **kwags):
#     old_stdout = sys.stdout # backup current stdout
#     sys.stdout = open(os.devnull, "w")
#     fn(*args, **kwags)
#     sys.stdout = sys.__stdout__ #old_stdout # reset old stdout


# Use pandas apply:
#def df_apply_funcs(df, idx_i=None, idx_j=None, drop_mol=True, verbose=True):
#    """
#    Apply set of functions on SMILES column to get basic information about molecule.
#    """
#
#    df.loc[idx_i:idx_j, 'Mol'] = [GetMol(smi) for smi in df.loc[idx_i:idx_j, 'SMILES']]
#    
#    # Standard representations:
#    if verbose:
#        print('Generate standard representations...')
#    df.loc[idx_i:idx_j, 'Canon_SMILES'] = \
#        df.loc[idx_i:idx_j].apply(lambda row: Chem.MolToSmiles(row['Mol'], canonical=True), axis=1)
#    df.loc[idx_i:idx_j, 'Canon_tauto_SMILES'] = \
#        df.loc[idx_i:idx_j].apply(lambda row: canonicalise_tautomer(row['SMILES']), axis=1)
#    df.loc[idx_i:idx_j, 'InChI'] = \
#        df.loc[idx_i:idx_j].apply(lambda row: Chem.MolToInchi(row['Mol']), axis=1)
#    df.loc[idx_i:idx_j, 'InChIKey'] = \
#        df.loc[idx_i:idx_j].apply(lambda row: Chem.MolToInchiKey(row['Mol']), axis=1)
#    df.loc[idx_i:idx_j, ['InChIKey_2D', 'InChIKey_stereo', 'InChIKey_N?']] = \
#        df.loc[idx_i:idx_j].apply(lambda row: row['InChIKey'].split('-'), axis=1, result_type='expand')
#    # Check InChI:
#    df.loc[idx_i:idx_j, 'InChI_Check'] = \
#        df.loc[idx_i:idx_j].apply(lambda row: inchi_check(row['InChI']), axis=1)
#    
#    # All possible representations:
#    if verbose:
#        print('Generate all possible representations...')
#    df.loc[idx_i:idx_j, 'Possible_tauto'] = \
#        df.loc[idx_i:idx_j].apply(lambda row: GetPossibleTautomers(row['Mol']), axis=1)
#    df.loc[idx_i:idx_j, 'N_possible_tauto'] = \
#        df.loc[idx_i:idx_j, 'Possible_tauto'].apply(list).apply(len)
#    df.loc[idx_i:idx_j, 'Possible_stereo'] = \
#        df.loc[idx_i:idx_j].apply(lambda row: GetPossibleStereoisomers(row['Mol']), axis=1)
#    df.loc[idx_i:idx_j, 'N_possible_stereo'] = \
#        df.loc[idx_i:idx_j, 'Possible_stereo'].apply(list).apply(len)
#    df.loc[idx_i:idx_j, 'Possible_tauto_stereo'] = \
#        df.loc[idx_i:idx_j].apply(lambda row: 
#                                  GetPossibleTautomersStereoisomers(row['Mol']), axis=1)
#
#    # Get additional information about the molecule:
#    if verbose:
#        print('Calculate additional information and descriptors...')
#    df.loc[idx_i:idx_j, 'Disconnect'] = \
#        df.loc[idx_i:idx_j].apply(lambda row: len(re.findall(r"\.", row['SMILES'])), axis=1)
#    df.loc[idx_i:idx_j, 'Atoms'] = \
#        df.loc[idx_i:idx_j].apply(lambda row: list(set(at.GetSymbol() for at in row['Mol'].GetAtoms())), axis=1)
#    df.loc[idx_i:idx_j, 'OB_Ionisable?'] = \
#        df.loc[idx_i:idx_j].apply(lambda row: is_ionisable(row['SMILES'], method='OpenBabel'), axis=1)
#    df.loc[idx_i:idx_j, 'OE_Ionisable?'] = \
#        df.loc[idx_i:idx_j].apply(lambda row: is_ionisable(row['SMILES'], method='OpenEye'), axis=1)
##     df['Undefined_stereo'] = df.apply(lambda row: undefined_stereo(row['Mol']), axis=1)
##     df['Unspecified_stereocentres'] = df.apply(lambda row: unspecified_stereocentres(row['Mol']), axis=1)
#
#    # Useful descriptors:
#    df.loc[idx_i:idx_j, 'MolWt'] = \
#        df.loc[idx_i:idx_j].apply(lambda row: Chem.rdMolDescriptors.CalcExactMolWt(row['Mol']), axis=1)
#    df.loc[idx_i:idx_j, 'Charge'] = \
#        df.loc[idx_i:idx_j].apply(lambda row: Chem.rdmolops.GetFormalCharge(row['Mol']), axis=1)
#    df.loc[idx_i:idx_j, 'NumAtoms'] = \
#        df.loc[idx_i:idx_j].apply(lambda row: row['Mol'].GetNumAtoms(), axis=1)
#    df.loc[idx_i:idx_j, 'NumHeavyAtoms'] = \
#        df.loc[idx_i:idx_j].loc[idx_i:idx_j].apply(lambda row: row['Mol'].GetNumHeavyAtoms(), axis=1)
#    df.loc[idx_i:idx_j, 'NumRotBonds'] = \
#        df.loc[idx_i:idx_j].loc[idx_i:idx_j].apply(lambda row: Chem.rdMolDescriptors.CalcNumRotatableBonds(row['Mol']), axis=1)
#
#    # Drop Mol column:
#    if drop_mol:
#        df.drop(columns='Mol', inplace=True)


def apply_funcs(df, 
                idx_i=None, 
                idx_j=None, 
                smiles_col='SMILES', 
                drop_mol=True, 
                verbose=True):
    """
    Apply set of functions on SMILES column to get basic information about molecule.
    """

    df.loc[idx_i:idx_j, 'Mol'] = [GetMol(smi) for smi in df.loc[idx_i:idx_j, smiles_col]]

    # Standard representations:
    if verbose:
        print('Generate standard representations...')
    df.loc[idx_i:idx_j, 'Canon_SMILES'] = \
        [Chem.MolToSmiles(mol, canonical=True) for mol in df.loc[idx_i:idx_j, 'Mol']]
    df.loc[idx_i:idx_j, 'Canon_tauto_SMILES'] = \
        [canonicalise_tautomer(smi) for smi in df.loc[idx_i:idx_j, smiles_col]]
    df.loc[idx_i:idx_j, '2D_SMILES'] = \
        [Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False) 
         for mol in df.loc[idx_i:idx_j, 'Mol']]
    df.loc[idx_i:idx_j, 'InChI'] = \
        [Chem.MolToInchi(mol) for mol in df.loc[idx_i:idx_j, 'Mol']]
    df.loc[idx_i:idx_j, 'InChIKey'] = \
        [Chem.MolToInchiKey(mol) for mol in df.loc[idx_i:idx_j, 'Mol']]
    df.loc[idx_i:idx_j, ['InChIKey_2D', 'InChIKey_stereo', 'InChIKey_N?']] = \
        [inchikey.split('-') for inchikey in df.loc[idx_i:idx_j, 'InChIKey']]

    # Check InChI:
    df.loc[idx_i:idx_j, 'InChI_Check'] = \
        [inchi_check(inchi) for inchi in df.loc[idx_i:idx_j, 'InChI']]

    # All possible representations and enantiomer if chiral:
    if verbose:
        print('Generate all possible representations...')
    df.loc[idx_i:idx_j, 'Enantiomer_SMILES'] = \
        df.loc[idx_i:idx_j].apply(lambda row: GetEnantiomer(row['Canon_SMILES']), axis=1)
    if len(df.loc[idx_i:idx_j]) == 1:
        df.loc[idx_i:idx_j, 'Possible_tauto'] = \
            df.loc[idx_i:idx_j].apply(lambda row: GetPossibleTautomers(row['Mol']), axis=1)
        df.loc[idx_i:idx_j, 'N_possible_tauto'] = \
            df.loc[idx_i:idx_j, 'Possible_tauto'].apply(list).apply(len)
        df.loc[idx_i:idx_j, 'Possible_stereo'] = \
            df.loc[idx_i:idx_j].apply(lambda row: GetPossibleStereoisomers(row['Mol']), axis=1)
        df.loc[idx_i:idx_j, 'N_possible_stereo'] = \
            df.loc[idx_i:idx_j, 'Possible_stereo'].apply(list).apply(len)
        df.loc[idx_i:idx_j, 'Possible_tauto_stereo'] = \
            df.loc[idx_i:idx_j].apply(lambda row: 
                                      GetPossibleTautomersStereoisomers(row['Mol']), axis=1)
    else:
        df.loc[idx_i:idx_j, 'Possible_tauto'] = \
            [list(GetPossibleTautomers(mol)) for mol in df.loc[idx_i:idx_j, 'Mol']]
        df.loc[idx_i:idx_j, 'N_possible_tauto'] = \
            [len(ls) for ls in df.loc[idx_i:idx_j, 'Possible_tauto']]
        df.loc[idx_i:idx_j, 'Possible_stereo'] = \
            [GetPossibleStereoisomers(mol) for mol in df.loc[idx_i:idx_j, 'Mol']]
        df.loc[idx_i:idx_j, 'N_possible_stereo'] = \
            [len(ls) for ls in df.loc[idx_i:idx_j, 'Possible_stereo']]
            #df.loc[idx_i:idx_j, 'Possible_stereo'].apply(list).apply(len)
        df.loc[idx_i:idx_j, 'Possible_tauto_stereo'] = \
            [GetPossibleTautomersStereoisomers(mol) for mol in df.loc[idx_i:idx_j, 'Mol']]

    # Get additional information about the molecule:
    if verbose:
        print('Calculate additional information and descriptors...')
    df.loc[idx_i:idx_j, 'Disconnect'] = \
        [len(re.findall(r"\.", smi)) for smi in df.loc[idx_i:idx_j, smiles_col]]
    if len(df.loc[idx_i:idx_j]) == 1:
        df.loc[idx_i:idx_j, 'Atoms'] = \
            df.loc[idx_i:idx_j].apply(lambda row: list(set(at.GetSymbol() for at in row['Mol'].GetAtoms())), axis=1)
    else:
        df.loc[idx_i:idx_j, 'Atoms'] = \
            [list(set(at.GetSymbol() for at in mol.GetAtoms())) for mol in df.loc[idx_i:idx_j, 'Mol']]
    df.loc[idx_i:idx_j, 'OB_Ionisable?'] = \
        [is_ionisable(smi, method='OpenBabel') for smi in df.loc[idx_i:idx_j, smiles_col]]
    df.loc[idx_i:idx_j, 'OE_Ionisable?'] = \
        [is_ionisable(smi, method='OpenEye') for smi in df.loc[idx_i:idx_j, smiles_col]]
#     df['Undefined_stereo'] = df.apply(lambda row: undefined_stereo(row['Mol']), axis=1)
#     df['Unspecified_stereocentres'] = df.apply(lambda row: unspecified_stereocentres(row['Mol']), axis=1)

    # Useful descriptors:
    df.loc[idx_i:idx_j, 'MolWt'] = \
        [Chem.rdMolDescriptors.CalcExactMolWt(mol) for mol in df.loc[idx_i:idx_j, 'Mol']]
    df.loc[idx_i:idx_j, 'Charge'] = \
        [Chem.rdmolops.GetFormalCharge(mol) for mol in df.loc[idx_i:idx_j, 'Mol']]
    df.loc[idx_i:idx_j, 'NumAtoms'] = \
        [mol.GetNumAtoms() for mol in df.loc[idx_i:idx_j, 'Mol']]
    df.loc[idx_i:idx_j, 'NumHeavyAtoms'] = \
        [mol.GetNumHeavyAtoms() for mol in df.loc[idx_i:idx_j, 'Mol']]
    df.loc[idx_i:idx_j, 'NumRotBonds'] = \
        [Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) for mol in df.loc[idx_i:idx_j, 'Mol']]

    # Drop Mol column:
    if drop_mol:
        df.drop(columns='Mol', inplace=True)


def run_apply_funcs(df, 
                    chunksize=None, 
                    smiles_col='SMILES', 
                    verbose=False):
    """
    Run apply_funcs() function, optionally in chunks, to get information on a 
    full dataset and also check for relationships between molecules (e.g. 
    enantiomers, stereoisomers, possible tautomers).
    """

    if chunksize is None:
        apply_funcs(df,
                    smiles_col=smiles_col,
                    verbose=verbose)
    else:
        for i in range(0, len(df), chunksize):
            end = '\r'
            if verbose:
                end = '\n'
            print('Chunk: {}/{}'.format((i//chunksize)+1,
                                        math.ceil(len(df)/chunksize)),
                  end=end)
            idx_i = df.index[i]
            j = i + chunksize
            if j > len(df):
                idx_j = None
            else:
                idx_j = df.index[j]
            apply_funcs(df, idx_i, idx_j, 
                        smiles_col=smiles_col, 
                        verbose=verbose)
        if not verbose:
            print()
    df['Enantiomer'] = find_enantiomers(df[['Canon_SMILES', 'Enantiomer_SMILES']])
    df['Stereoisomers'] = find_stereoisomers(df['2D_SMILES'])
    check_for_duplicates(df, verbose=True)


# Identify possible duplicates:
def check_for_duplicates(df, verbose=True):
    """
    Check for compounds which may be duplicates (as tautomers or due to 
    undefined stereochemistry).
    """

    if verbose:
        print('Identify possible duplicates...')
    df['Possible_tauto_duplicates'] = find_possible_duplicates(df['Possible_tauto'])
    df['Possible_stereo_duplicates'] = find_possible_duplicates(df['Possible_stereo'])
    df['Possible_tauto_stereo_duplicates'] = find_possible_duplicates(df['Possible_tauto_stereo'])


def generate_dataset_report(df):
    """
    Generate a set of plots to summarise the dataset.
    """
    print('Number of datapoints: {}'.format(len(df)))
    print('Number of unique canonical SMILES: {}'.format(len(df.drop_duplicates('Canon_SMILES'))))
    print('Number of unique 2D canonical SMILES: {}'.format(len(df.drop_duplicates('2D_SMILES'))))
    print('Number of enantiomer pairs: {}'.format(len(df['Enantiomer'].drop_na())//2))
