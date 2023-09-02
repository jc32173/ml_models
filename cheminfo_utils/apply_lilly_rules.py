import subprocess
import tempfile
import pandas as pd
import glob
from rdkit import Chem

import sys
sys.path.insert(0, '/users/xpb20111/programs/')
from python_utils.pandas_utils import conv_df_to_str


# Probably move this to smi_funcs.py:
def KekulizeSmiles(smi):
    mol = Chem.MolFromSmiles(smi)
    Chem.Kekulize(mol)
    return Chem.MolToSmiles(mol, kekuleSmiles=True)


# Eventually make this a wrapper around run_lilly_rules() to make it more user friendly at the 
# expense of some performance:
def apply_lilly_rules(df=None,
                      smiles_col='SMILES',
                      smiles=[],
                      ids=None,
                      smi_input_filename=None,
                      cleanup=True,
                      run_in_temp_dir=True,
                      lilly_rules_script=\
                          '/users/xpb20111/software/'+\
                          'Lilly-Medchem-Rules/Lilly_Medchem_Rules.rb'):
    """
    Apply Lilly rules to SMILES in a list or a DataFrame.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame containing SMILES
    smiles_col: str
        Name of SMILES column

    Returns
    -------
    pd.DataFrame
        DataFrame containing results of applying Lilly's rules to SMILES, including pass/fail and warnings

    Example
    -------
    >>> apply_lilly_rules(smiles=['CCCCCCC(=O)O', 'CCC', 'CCCCC(=O)OCC', 'c1ccccc1CC(=O)C'])
                SMILES       SMILES_Kekule  Lilly_rules_pass      Lilly_rules_warning  Lilly_rules_SMILES
    0     CCCCCCC(=O)O        CCCCCCC(=O)O              True        D(80) C6:no_rings        CCCCCCC(=O)O
    1              CCC                 CCC             False     TP1 not_enough_atoms                 CCC
    2     CCCCC(=O)OCC        CCCCC(=O)OCC              True  D(75) ester:no_rings:C4        CCCCC(=O)OCC
    3  c1ccccc1CC(=O)C  CC(=O)CC1=CC=CC=C1              True                     None  CC(=O)CC1=CC=CC=C1
    """

    # Write from DataFrame:
    if df is not None:
        if isinstance(df, pd.core.series.Series):
            df = df.to_frame()

        df.insert(0, 'ID', df.index)

    # Write from list of SMILES:
    else:
        if isinstance(smiles, str):
            smiles = [smiles]

        # Need IDs to identify SMILES after passing through Lilly rules script
        if ids is None:
            ids = [str(i) for i in range(len(smiles))]
            
        df = pd.DataFrame(data=zip(ids, smiles), 
                          columns=['ID', 'SMILES'], 
                          index=ids)

        # temp.write('\n'.join([f'{i} {j}' for i, j in zip(smiles, ids)]))

    n_cmpds = len(df)

    df[smiles_col+'_Kekule'] = df[smiles_col].apply(KekulizeSmiles)

    smi_file_txt = conv_df_to_str(df[[smiles_col+'_Kekule', 'ID']],
                                  sep=' ',
                                  header=False,
                                  index=False)

    # Optionally set up temporary directory:
    if run_in_temp_dir:
        temp_dir = tempfile.TemporaryDirectory()
        run_dir = temp_dir.name + '/'
    else:
        run_dir = './'

    # If filename given, save SMILES to this file:
    if smi_input_filename is not None:
        with open(run_dir+smi_input_filename, 'w') as temp:
            temp.write(smi_file_txt)

    # If no filename given just use a temporary file:
    else:
        # Lilly rules script reads the file suffix so needs to be .smi:
        temp = tempfile.NamedTemporaryFile(mode="w+", 
                                           suffix=".smi", 
                                           dir=run_dir)
        temp.write(smi_file_txt)
        # Go to start of file:
        temp.seek(0)

    # Run Lilly rules script
    lilly_results = \
            subprocess.run([f'cd {run_dir}; {lilly_rules_script} {temp.name}'], 
                   shell=True, 
                   stdout=subprocess.PIPE, 
                   stderr=subprocess.PIPE)

    if lilly_results.stderr.decode('utf-8') != '':
        print('WARNING: {}'.format(lilly_results.stderr.decode('utf-8'))) 
    lilly_results = lilly_results.stdout.decode('utf-8')
   
    # Process results:
    passes = []
    if lilly_results != '':
        for line in lilly_results.strip().split('\n'):

            # Record warning if given:
            if ' : ' in line:
                smiles_molid, warning = line.split(' : ')
            else:
                smiles_molid = line.strip()
                warning = None
            smiles, molid = smiles_molid.split(' ')
            passes.append([molid, warning, smiles])

    # Get reasons for failures:
    failures = []
    for bad_file in glob.glob(run_dir+'bad*.smi'):
        for line in open(bad_file, 'r').readlines():
            line = line.split(' ')
            smiles = line[0]
            molid = line[1]
            warning = ' '.join(line[2:]).strip(': \n')
            failures.append([molid, warning, smiles])

    # Close and remove tempfile:
    # (Do this even if run in a temporary directory to prevent warning when
    # script finishes and tries to remove temporary file at that point)
    if smi_input_filename is None:
        temp.close()

    if run_in_temp_dir:
        temp_dir.cleanup()
    elif cleanup:
        subprocess.run(['rm -f ok{0,1,2,3}.log bad{0,1,2,3}.smi'], shell=True)

    # Convert to DataFrame:
    df_passes = pd.DataFrame(passes, 
                             columns=['ID', 
                                      'Lilly_rules_warning', 
                                      'Lilly_rules_SMILES'])
    #                .set_index('ID', verify_integrity=True)
    df_passes.insert(0, 'Lilly_rules_pass', True)

    df_failures = pd.DataFrame(failures, 
                               columns=['ID', 
                                        'Lilly_rules_warning', 
                                        'Lilly_rules_SMILES'])
    #                .set_index('ID', verify_integrity=True)
    df_failures.insert(0, 'Lilly_rules_pass', False)

    df_all = pd.concat([df_passes, df_failures], axis=0)\
               .set_index('ID', verify_integrity=True)

    df_out = pd.merge(left=df.drop(columns='ID'), left_index=True, 
                      right=df_all, right_index=True, 
                      how='inner')

    # Check all molecules accounted for:
    if len(df_out) != len(df):
        raise ValueError('Some compounds missing, {} molecules input, but {} compounds output.'.format(len(df), len(df_out)))

    #df['Lilly_rules_pass'].fillna(False, inplace=True)

    return df_out
