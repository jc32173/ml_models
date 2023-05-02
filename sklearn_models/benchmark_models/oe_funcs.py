# OpenEye functions, to be combined with smi_funcs file eventually

import openeye as oe
from openeye import oechem

def oe_reasonable_tautomer(smi, ph=True):
    # Error handling?

    mol = oechem.OEGraphMol()

    echem.OESmilesToMol(mol, smi)

    oe.OEGetReasonableProtomer(mol, pka)

    smi = oechem.OEMolToSmiles(mol)

    return smi

# Using OEGetUniqueProtomer
def canonicalise_tautomer(smi)

    #try:
    #    import openeye as oe
    #    from openeye import oechem

    mol = oechem.OEGraphMol()
    canon_tauto_mol = oechem.OEGraphMol()

    echem.OESmilesToMol(mol, smi)
    oe.OEGetUniqueProtomer(canon_tauto_mol, mol)

    canon_tauto_smi = oechem.OEMolToSmiles(mol)

    return canon_tauto_smi

# Using OEEnumerateTautomers
def canonicalise_tautomer(smi)

    #try:
    #    import openeye as oe
    #    from openeye import oechem

    mol = oechem.OEGraphMol()
    canon_tauto_mol = oechem.OEGraphMol()

    echem.OESmilesToMol(mol, smi)
    oe.OEEnumerateTautomers(canon_tauto_mol, mol)

    canon_tauto_smi = oechem.OEMolToSmiles(mol)

    return canon_tauto_smi

def adjust_for_ph(smi):
    # Error handling?

    mol = oechem.OEGraphMol()

    echem.OESmilesToMol(mol, smi)

    oe.OESetNeutralpHModel(mol)

    smi = oechem.OEMolToSmiles(mol)

    return mol
