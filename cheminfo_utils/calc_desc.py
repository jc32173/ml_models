# Writen October 2020

import sys

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors

import numpy as np
import pandas as pd
import re

from cheminfo_utils.smi_funcs import canonicalise_tautomer, correct_smiles, \
    adjust_for_ph, calc_rdkit_descs


# Process SMILES before calculating descriptors 
# (canonicalise tautomer, correct smiles, adjust for ph)
def process_smiles(smi, tauto=False, ph=None, phmodel=None): #, return_mol=True):
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


def calc_desc(smi_ls,
              #id_ls=True,
              tauto=True,
              ph=False,
              phmodel=None,
              descriptors=['RDKit'],
              fingerprints=[],
              morgan_fps_opts={'radius' : 4},
              rm_na=True,
              rm_const=False,
              output_processed_smiles=False):
    """
    Calculate descriptors/fingerprints.
    """

    if isinstance(smi_ls, str):
        smi_ls = [smi_ls]

    #if isinstance(smi_ls, list):
    #    df_desc = pd.DataFrame(data=smi_ls, columns=['SMILES'])
    #    smi_col = 'SMILES'

    if isinstance(smi_ls, pd.core.frame.DataFrame):
        smi_ls = smi_ls.squeeze()

    if isinstance(smi_ls, pd.core.series.Series):
        smi_ls = smi_ls.to_list()
    #    df_desc = pd.DataFrame(smi_ls)
    #    smi_col = smi_ls.name

    processed_smiles = [process_smiles(smi,
                                       tauto=tauto,
                                       ph=ph,
                                       phmodel=phmodel)[0] 
                        for smi in smi_ls]

    mols = [Chem.MolFromSmiles(smi) for smi in processed_smiles]

    descs = []

    # Descriptors:

    if 'RDKit' in descriptors:
        if isinstance(descriptors, dict):
            desc_ls = descriptors['RDKit']
        else:
            desc_ls = []

        rdkit_desc = calc_rdkit_descs(mols, id_ls=smi_ls, desc_ls=desc_ls)[0]
        rdkit_desc.columns = ['RDKit_'+col for col in rdkit_desc.columns]
        descs.append(rdkit_desc)

    if 'Mordred' in descriptors:
        raise NotImplementedError('')
        #if isinstance(descriptors, dict):
        #    desc_ls = descriptors['Mordred']
        #else:
        #    desc_ls = None

        #descs.append(calc_mordred_desc(processed_smiles))

    if 'CDDD' in descriptors:
        # See: https://github.com/jrwnter/cddd
        raise NotImplementedError('')

    # Fingerprints:

    if 'RDKit' in fingerprints:

        rdkit_fps = np.array([Chem.RDKFingerprint(mol).ToList() 
                              for mol in mols])
        rdkit_fps = pd.DataFrame(data=rdkit_fps, 
                                 index=smi_ls, 
                                 columns=['RDKit_fp_b'+str(i) 
                                          for i in range(rdkit_fps.shape[1])])
        descs.append(rdkit_fps)

    if 'Morgan' in fingerprints:
        morgan_fps = np.array([
            AllChem.GetMorganFingerprintAsBitVect(mol, 
                                                  **morgan_fps_opts)
            for mol in mols])
        morgan_fps = pd.DataFrame(data=morgan_fps, 
                                  index=smi_ls,
                                  columns=['Morgan_fp_b'+str(i)
                                           for i in range(morgan_fps.shape[1])])
        descs.append(morgan_fps)
    
    df_desc = pd.concat(descs, axis=1)

    # Check no NaN in descriptors and remove if found:
    if rm_na:
        if df_desc.isnull().values.any():
            print('WARNING: Initial set of descriptors contained NaN values:')
            print('Descriptor\tNumber of molecules with NaN values')
            for d_nan in df_desc_only.columns[df_desc_only.isnull().any()]:
                print(d_nan+'\t'+sum(df_desc_only[d_nan].isna()))
            print('These will be removed')
            df_desc = df_desc.drop(df_desc.columns[df_desc.isnull().any()], axis=1, inplace=True)

    # Remove constant descriptors?

    if output_processed_smiles:
        pass

    return df_desc

