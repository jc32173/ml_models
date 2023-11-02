#!/bin/env python

# November 2022
# Functions for calculating predictions and descriptors for a new dataset.

import pandas as pd
import numpy as np
import sys, os
from datetime import datetime
import pathlib
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from openeye import oechem as oe
from openeye import oemolprop as mp
from openeye import oeomega
from openeye import oequacpac

#sys.path.insert(0, '/users/xpb20111/repos/PP/')
from cheminfo_utils.apply_lilly_rules import apply_lilly_rules


def class_sol(sol):
    """
    Convert solubility model output to class.
    """
    if sol == 2:
        return 'High'
    elif sol == 1:
        return 'Medium'
    elif sol == 0:
        return 'Low'
    else:
        return sol


def error_wrapper(smi, mod):
    """
    Wrapper to catch errors from the prediction models
    for certain molecules (e.g. molecules containing 
    certain atoms such as I).
    """
    try:
        return mod(smi)
    except ValueError:
        return None


def canonicalise_SMILES(smi):
    """
    Canonicalise SMILES using RDKit.
    """
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi))


def canonicalise_tautomer(mol):
    """
    Canonicalise tautomer using RDKit.
    """
    enumerator = rdMolStandardize.TautomerEnumerator()
    # enumerator.SetReassignStereo = True
    # enumerator.SetRemoveBondStereo = False
    # enumerator.SetRemoveSp3Stereo = False
    mol = enumerator.Canonicalize(mol)
    return mol


def read_into_oemol(mol_str, mol, read_InChI=False):
    """
    Convert InChI or SMILES to OEMol object.
    """
    if read_InChI:
        read_success = oe.OEInChIToMol(mol, mol_str)
    else:
        read_success = oe.OESmilesToMol(mol, mol_str)
    return read_success


def CalcLogP_OE(mol_str_ls, read_InChI=False):
    """
    Calculate OpenEye logP descriptor for a list of SMILES or InChIs.
    """

    if isinstance(mol_str_ls, str):
        mol_str_ls = [mol_str_ls]

    logp_ls = []

    mol = oe.OEGraphMol()
    for mol_str in mol_str_ls:
        if not read_into_oemol(mol_str, mol, read_InChI=read_InChI):
            print('ERROR: Cannot read molecule: {} into OpenEye.'.format(mol_str))
            logp_ls.append(np.nan)
        else:
            logp_ls.append(mp.OEGetXLogP(mol, atomxlogps=None))
    return logp_ls


def substruct_match(mol, substructs, canonicalise_tauto=False):
    """
    Match substructures.
    """
    matches = []
    if canonicalise_tauto:
        mol = canonicalise_tautomer(mol)
    mol = Chem.AddHs(mol)
    for substruct in substructs:
        matches.append(mol.HasSubstructMatch(substruct))
    return matches


def enumerate_tautomers_isomers(mol_str_ls, 
                                enum_tauto=True, 
                                enum_iso=True, 
                                canonicalise_smi=False, 
                                read_InChI=False):
    """
    Enumerate all tautomers and stereoisomers for a list of SMILES or InChIs.
    Based on FRED docking script.
    """

    if isinstance(mol_str_ls, str):
        mol_str_ls = [mol_str_ls]

    full_smiles_ls = []

    # Tautomer options:
    tautomer_opts = oequacpac.OETautomerOptions()
    tautomer_opts.SetMaxSearchTime(300)
    tautomer_opts.SetRankTautomers(True)
    tautomer_opts.SetCarbonHybridization(False)

    # Stereoisomer options:
    flipper_opts = oeomega.OEFlipperOptions()
    flipper_opts.SetMaxCenters(12)
    flipper_opts.SetEnumSpecifiedStereo(False)
    #flipper_opts.SetEnumNitrogen(True)
    flipper_opts.SetWarts(False)

    # Get oemol:
    mol = oe.OEGraphMol()
    for mol_str in mol_str_ls:
        full_smiles_ls.append([])
        if not read_into_oemol(mol_str, mol, read_InChI=read_InChI):
            print('ERROR: {}'.format(mol_str))
            continue
        else:
            # Generate tautomers and stereoisomers
            tautomers = []
            if enum_tauto:
                for tautomer in oequacpac.OEEnumerateTautomers(mol, tautomer_opts):
                #for tautomer in oequacpac.OEGetReasonableTautomers(mol, tautomer_opts, pKa_norm):
                    tautomers.append(tautomer)
            else:
                tautomers.append(mol)
            stereoisomers = []
            if enum_iso:
                for tautomer in tautomers:
                    for stereoisomer in oeomega.OEFlipper(tautomer, flipper_opts):
                        stereoisomers.append(stereoisomer)
            else:
                stereoisomers = tautomers
            for out_mol in stereoisomers:
                full_smiles_ls[-1].append(oe.OEMolToSmiles(out_mol))
    if canonicalise_smi:
        full_smiles_ls = [canonicalise_SMILES(smi) for smi in full_smiles_ls]
    return full_smiles_ls


# Calculate histogram:
def calc_hist(data, bin_width):
    """
    Histogram data.
    """
    min_bin = (min(data)//bin_width)*bin_width
    max_bin = (max(data)//bin_width)*bin_width
    # Maximum of bins must be bin_width*2 to ensure 
    # right bin edge is included:
    hist, bins = np.histogram(data, 
                              bins=np.arange(min_bin, 
                                             max_bin+(bin_width*2), 
                                             bin_width))
    # Need to make bins integers so that histograms can be summed:
    bins = np.round(bins/bin_width).astype(int)
    return hist, bins


def time_wrapper(fn, inputs):
    """
    Wrapper to run function and record time taken.
    """
    start_time = datetime.now()
    out = fn(inputs)
    end_time = datetime.now()
    time_taken = end_time - start_time 
    return out, time_taken.total_seconds()


def calc_predictions(df_all_vals, 
                     #smiles, 
                     #mols, 
                     save_hists={'pIC50_pred' : 0.1, 
                                 'MPO' : 0.1}, 
                     models={}, 
                     descriptors={'molwt' : \
                                    Chem.rdMolDescriptors.CalcExactMolWt, 
                                  'n_aromatic_rings' : \
                                    Chem.rdMolDescriptors.CalcNumAromaticRings}, 
                     substructs=None, 
                     hist_by_substruct=False, 
                     calc_logp_oe=True, 
                     calc_pfi=True, 
                     calc_mpo=True, 
                     write_header=True, 
                     outfile='', 
                     all_hists={}, 
                     substruct_hists={}):
    """
    Calculate predictions and descriptors for a list of SMILES.
    """

    start_time = datetime.now()

    #df_all_vals = pd.DataFrame(data=[], index=smiles)
    n_smiles = len(df_all_vals['SMILES'])

    for mod_name in models['_order']:
        mod = models[mod_name]
        # Save all values from models with multiple outputs:
        if isinstance(mod_name, tuple):
            mod_name = list(mod_name)
        df_all_vals[mod_name], pred_time = time_wrapper(mod, df_all_vals['SMILES'])
        # Label time taken column based on the name of first 
        # output for models with multiple outputs:
        if isinstance(mod_name, list):
            mod_name = mod_name[0]
        df_all_vals[mod_name+'_time'] = [pred_time/n_smiles]*n_smiles

    #mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    for desc_name in descriptors['_order']:
        desc = descriptors[desc_name]
        df_all_vals[desc_name] = np.array([desc(mol) for mol in df_all_vals['Mol']])

    if calc_logp_oe:
        df_all_vals['LogP_OE'] = np.array([CalcLogP_OE(smi) for smi in df_all_vals['SMILES']])

    if calc_pfi:
        df_all_vals['PFI'] = df_all_vals['n_aromatic_rings'] + df_all_vals['LogP_OE']

    if calc_mpo:
        df_all_vals['MPO'] = \
                (-df_all_vals['pIC50_pred'])*(1/(1 + np.exp(df_all_vals['PFI'] - 8)))

    if substructs is not None:
        df_all_vals[substructs['substruct_name']] = \
            np.array([substruct_match(mol, substructs['substruct_mol']) #, canonicalise_tauto=True)
                      for mol in df_all_vals['Mol']])

    end_time = datetime.now()
    df_all_vals['av_tot_time'] = (end_time - start_time).total_seconds()/n_smiles

    # Calculate histograms:
    for val_name, bin_width in save_hists.items():
        if val_name not in df_all_vals.columns:
            raise ValueError('Cannot histogram {} as the value was not calculated.'.format(val_name))
        hist, bins = calc_hist(df_all_vals[val_name], bin_width)
        all_hists[val_name] = \
                all_hists[val_name].add(pd.Series(data=hist, #.astype(int), 
                                                  index=bins[:-1], 
                                                  name=val_name+'_hist'), 
                                        fill_value=0).astype(int)
        #all_hists[val_name].index.rename('bin', inplace=True)

        if hist_by_substruct:
            for substruct_name in substructs['substruct_name']:
                if len(df_all_vals.loc[df_all_vals[substruct_name]]) > 0:

                    hist, bins = calc_hist(df_all_vals.loc[df_all_vals[substruct_name]][val_name], bin_width)

                    # Add new row for bins not already in dataframe:
                    substruct_hists[val_name] = \
                    substruct_hists[val_name].append(pd.DataFrame(columns=substruct_hists[val_name].columns, 
                                                                  index=list(set(bins[:-1]) - set(substruct_hists[val_name].index))))\
                                             .fillna(0)\
                                             .astype(int)\
                                             .sort_index()

                    substruct_hists[val_name].loc[bins[:-1], val_name+'_'+substruct_name+'_hist'] = \
                    substruct_hists[val_name].loc[bins[:-1], val_name+'_'+substruct_name+'_hist']\
                                             .add(pd.Series(data=hist, #.astype(int),
                                                            index=bins[:-1],
                                                            name=val_name+'_'+substruct_name+'_hist'),
                                                  fill_value=0).astype(int)

    # Output values:
    if outfile != '':
        df_all_vals.drop('Mol', axis=1)\
                   .to_csv(outfile, mode='a', sep=';', header=write_header)
    else:
        return df_all_vals.drop('Mol', axis=1)


def process_df(infile, 
               sep=';', 
               structure_col='InChI', 
               start_line=0, 
               end_line=-1, 
               index_col=False, 
               index_prefix='', 
               #compression='gzip', 
               chunksize=1000, 
               invalid_inchi_file=None, 
               canonicalise_tauto=True, 
               enum_tauto=False, 
               enum_iso=False, 
               lilly_rules=False, 
               lilly_rules_script='Lilly_Medchem_Rules.rb',
               drop_lilly_failures=False, 
               save_hists={'pIC50_pred' : 0.1, 
                           'MPO' : 0.1}, 
               models={}, 
               descriptors={'molwt' : \
                                rdMolDescriptors.CalcExactMolWt, 
                            'n_aromatic_rings' : \
                                rdMolDescriptors.CalcNumAromaticRings}, 
               substructs=None, 
               hist_by_substruct=False, 
               calc_logp_oe=True, 
               calc_pfi=True, 
               calc_mpo=True, 
               outfile='preds.csv', 
               hist_file_prefix='hist', 
               append_to_hist=True):
    """
    Run calc_predictions on a dataframe.
    """

    start_time = datetime.now()

    # Set up empty histograms:
    all_hists = {}
    substruct_hists = {}
    for val_name in save_hists.keys():
        all_hists[val_name] = pd.Series(dtype=int, 
                                        name=val_name+'_hist')
        #all_hists[val_name].index.rename('bin', inplace=True)
        if hist_by_substruct:
            substruct_hists[val_name] = \
            pd.DataFrame(dtype=int, 
                         columns=[val_name+'_'+str(substruct_name)+'_hist' 
                                  for substruct_name in substructs['substruct_name'].to_list()])

    if end_line == -1:
        nrows = None
    else:
        nrows = end_line - start_line

    n_mols = 0

    # Read input file into dataframe in chunks:
    infile_ext = pathlib.Path(infile).suffixes
    if '.inchi' in infile_ext:
        df_iter = pd.read_csv(infile, 
                              #compression=compression, 
                              chunksize=chunksize, 
                              sep=' ',
                              header=None,
                              usecols=[0],
                              names=[structure_col], 
                              skiprows=start_line, 
                              nrows=nrows)

    elif '.csv' in infile_ext:
        df_iter = pd.read_csv(infile,
                              chunksize=chunksize,
                              sep=sep,
                              header=0,
                              usecols=[structure_col],
                              skiprows=range(1, start_line+1),
                              nrows=nrows)
    else:
        raise ValueError('Cannot recognise file extension, '+\
            'file must be .csv or .inchi (with .gz if compressed).')

    for df_i, df in enumerate(df_iter):

        if (df_i == 0) and (df[structure_col].iloc[0][:5] != 'InChI'):
            raise ValueError(
                'Currently code only reads molecule structures from InChIs.')

        # Create an index column if one doesn't already exist:
        if not index_col:
            df.index = df.index + start_line
            if index_prefix != '':
                df['InChI_ID'] = 'i' + df.index.astype(str)
                df.index = index_prefix + '-' + df['InChI_ID'].astype(str)
            df.index.rename('ID', inplace=True)
        else:
            df.set_index(index_col, verify_integrity=True, inplace=True)

        # Enumerate over tautomers or stereoisomers:
        if enum_tauto or enum_iso:
            if canonicalise_tauto:
                # Ensure inchi is a string here to catch any missing inchis or nan 
                # values, these will be picked up on next line:
                df['Mol'] = [Chem.MolFromInchi(str(inchi)) for inchi in df[structure_col]]

                # Remove any InChIs which cannot be read into RDKit:
                if invalid_inchi_file is not None:
                    df.loc[df['Mol'].isna(), structure_col]\
                      .to_csv(invalid_inchi_file, sep=sep, mode='a', header=False)
                df = df.loc[df['Mol'].notna()]

                df['Mol'] = [canonicalise_tautomer(mol) for mol in df['Mol']]
                df['canon_tauto_SMILES'] = [Chem.MolToSmiles(smi) for smi in df['Mol']]
                tauto_iso_smiles = enumerate_tautomers_isomers(df['canon_tauto_SMILES'], 
                                                               enum_tauto=enum_tauto, 
                                                               enum_iso=enum_iso, 
                                                               canonicalise_smi=False, 
                                                               read_InChI=False)
            else:
                tauto_iso_smiles = enumerate_tautomers_isomers(df[structure_col], 
                                                               enum_tauto=enum_tauto, 
                                                               enum_iso=enum_iso, 
                                                               canonicalise_smi=False, 
                                                               read_InChI=True)
            # Add new index to take account of SMILES:
            df_smi = pd.Series(tauto_iso_smiles, 
                               index=df.index, 
                               name='SMILES')\
                       .map(enumerate).map(list).explode()
            df = pd.merge(left=df, 
                          left_index=True, 
                          right=pd.DataFrame(df_smi.to_list(), 
                                             columns=['SMILES_ID', 'SMILES'], 
                                             index=df_smi.index), 
                          right_index=True)
            df['SMILES_ID'] = 's' + df['SMILES_ID'].astype(str)
            df['ID'] = df.index.astype(str) + '.' + df['SMILES_ID'].astype(str)
            df.set_index('ID', verify_integrity=True, inplace=True)
            df['Mol'] = [Chem.MolFromSmiles(smi) for smi in df['SMILES']]

        else:
            # Ensure inchi is a string here to catch any missing inchis or nan 
            # values, these will be picked up on next line:
            df['Mol'] = [Chem.MolFromInchi(str(inchi)) for inchi in df[structure_col]]

            # Remove any InChIs which cannot be read into RDKit:
            if invalid_inchi_file is not None:
                df.loc[df['Mol'].isna(), structure_col]\
                  .to_csv(invalid_inchi_file, sep=sep, mode='a', header=False)
            df = df.loc[df['Mol'].notna()]

            if canonicalise_tauto:
                df.loc[:,'Mol'] = [canonicalise_tautomer(mol) for mol in df['Mol']]
            df.loc[:,'SMILES'] = [Chem.MolToSmiles(mol) for mol in df['Mol']]

        # Apply Lilly rules here in case decide to drop compounds which don't pass:
        if lilly_rules:
            if not os.path.isfile(lilly_rules_script):
                print("WARNING: Cannot find Lilly rules script "+\
                      "(Lilly_Medchem_Rules.rb) at: "+lilly_rules_script+\
                      ", will skip Lilly rules.")
                lilly_rules = False
            else:
                df = apply_lilly_rules(df=df,
                                       smiles_col='SMILES', 
                                       run_in_temp_dir=True, 
                                       lilly_rules_script=lilly_rules_script)
                if drop_lilly_failures:
                    df = df.drop(~df['Lilly_rules_pass'])

        if len(df['SMILES']) == 0:
            print('WARNING: No molecules in chunk {} for predictions'.format(df_i)+\
                  ' (molecules may be invalid or fail Lilly rules (if --drop_lilly_failures)')
            continue

        n_mols += len(df['SMILES'])

        # If enumerated over tautomers or stereoisomers still 
        # make predictions in chunks close to chunksize:
        for enum_i, df in enumerate(np.array_split(df, max([int(round(len(df)/chunksize)), 1]))):
            calc_predictions(df, 
                             #smiles=df['SMILES'], 
                             #mols=df['Mol'], 
                             save_hists=save_hists, 
                             models=models, 
                             descriptors=descriptors, 
                             substructs=substructs, 
                             hist_by_substruct=hist_by_substruct, 
                             calc_logp_oe=calc_logp_oe, 
                             calc_pfi=calc_pfi, 
                             calc_mpo=calc_mpo, 
                             write_header=not bool(df_i) and not bool(enum_i), 
                             outfile=outfile, 
                             all_hists=all_hists, 
                             substruct_hists=substruct_hists)


    # Save histograms:
    for val_name, bin_width in save_hists.items():
        hist_outfile = hist_file_prefix.split('.')[0].strip('_')+'_'+val_name+'.csv'
        hist = all_hists[val_name]

        # Convert integer bins to correct values:
        hist.index = hist.index*bin_width
 
        # Round bin edges to correct number of dp:
        n_dp = 0
        if '.' in str(bin_width):
            n_dp = len(str(bin_width).split('.')[1])
            hist.index = np.round(hist.index, n_dp)

        # Read in previous histogram data if appending:
        # (Could use blocking in case jobs are running in parallel and trying 
        # to write to the same file)
        if append_to_hist and os.path.isfile(hist_outfile):
            prev_hist = pd.read_csv(hist_outfile, index_col=0).squeeze()
            hist = hist.add(prev_hist, fill_value=0).astype(int)

        # Fill in any missing bins with zeros:
        full_bin_range = np.round(np.arange(hist.index.min(), hist.index.max(), bin_width), n_dp)
        hist = hist.add(pd.Series(data=np.zeros(len(full_bin_range), dtype=int), 
                                  index=full_bin_range, 
                                  name=val_name+'_hist'), 
                        fill_value=0).astype(int)

        # Save histogram data:
        hist.index.rename('bin', inplace=True)
        hist.to_csv(hist_outfile)

    # Save substructure histograms:
    if hist_by_substruct:
        for val_name, bin_width in save_hists.items():
            hist_outfile = hist_file_prefix.split('.')[0].strip('_')+'_'+val_name+'_by_substructure.csv'
            hist = substruct_hists[val_name]

            # Convert integer bins to correct values:
            hist.index = hist.index*bin_width

            # Round bin edges to correct number of dp:
            n_dp = 0
            if '.' in str(bin_width):
                n_dp = len(str(bin_width).split('.')[1])
                hist.index = np.round(hist.index, n_dp)

            # Read in previous histogram data if appending:
            # (Could use blocking in case jobs are running in parallel and trying
            # to write to the same file)
            if append_to_hist and os.path.isfile(hist_outfile):
                prev_hist = pd.read_csv(hist_outfile, index_col=0).squeeze()
                hist = hist.add(prev_hist, fill_value=0).astype(int)

            # Fill in any missing bins with zeros:
            full_bin_range = np.round(np.arange(hist.index.min(), hist.index.max(), bin_width), n_dp)
            hist = hist.append(pd.DataFrame(columns=hist.columns, 
                                            index=list(set(full_bin_range) - set(hist.index))))\
                       .fillna(0)\
                       .astype(int)\
                       .sort_index()

            # Save histogram data:
            hist.index.rename('bin', inplace=True)
            hist.to_csv(hist_outfile)

    end_time = datetime.now()

    total_time = (end_time - start_time).total_seconds()
    print('Total time: {:.3f} s (~{:.3f} s/molecule)'.format(
        total_time, total_time/n_mols))
