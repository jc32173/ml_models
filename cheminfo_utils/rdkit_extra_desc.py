"""
Functions to calculate additional descriptors not available in RDKit.
Descriptors are calculated from RDKit mol objects.
"""


from rdkit import Chem
import numpy as np
import sys, os

# Should be able to get RDContribDir from: 
# from rdkit.Chem import RDConfig
# RDContribDir = RDConfig.RDContribDir
# but isn't pointing to the right directory in deepchem environment so hard-code instead:
RDContribDir = '/users/xpb20111/.conda/envs/deepchem/share/RDKit/Contrib/'
sys.path.append(os.path.join(RDContribDir, 'SA_Score'))
import sascorer


# Useful for checking for macrocycles:
def max_ring_size(mol):
    # From: https://sourceforge.net/p/rdkit/mailman/message/36781319/
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    ri = mol.GetRingInfo()
    max_ring_size = max((len(r) for r in ri.AtomRings()), default=0)
    return max_ring_size


# From: Troubleshoot_issues_with_acpype:
def GetTotalNumProtons(mol):
    n_proton = 0
    for atom in mol.GetAtoms():
        n_proton += atom.GetAtomicNum()
    return n_proton


# From: Troubleshoot_issues_with_acpype:
def GetTotalNumElectrons(mol):
    n_elec = Chem.GetFormalCharge(mol)*(-1)
    n_elec += GetTotalNumProtons(mol)
    return n_elec


def GetNumHs(mol, onlyExplicit=False):
    """
    Get number of hydrogens in a molecule
    """
    return mol.GetNumAtoms(onlyExplicit=onlyExplicit) - mol.GetNumHeavyAtoms()


# Originally written for 202211_Analyse_40M_pymolgen_compounds_ONGOING:
def GetFusedRings(cmpd):
    """
    Get the groups of fused rings in a molecule,
    ordered by size.

    Examples for testing the code:
    >>> GetFusedRings('Cc1noc(C)c1-c1cnc2[nH]c(=O)n(-c3c(C(N)=O)c4nc5ccccc5oc-4cc3=O)c2c1')
    [{3, 4, 5}, {1, 2}, {0}]
    >>> GetFusedRings('Cc1noc(C)c1-c1cnc2[nH]c(=O)n(Cn3c4ccccc4c4c5c(c6c7ccccc7[nH]c6c43)C(=O)NC5)c2c1')
    [{3, 4, 5, 6, 7, 8}, {1, 2}, {0}]
    >>> GetFusedRings('Cc1noc(C)c1-c1cnc2[nH]c(=O)n(-c3c4c(nc5cc(F)ccc35)CC3C=CCC4C3)c2c1')
    [{3, 4, 5, 6}, {1, 2}, {0}]
    >>> GetFusedRings('Cc1noc(C)c1-c1cnc2[nH]c(=O)n(Cc3ccc4c(c3)[nH]c3c4c4c(c5c6cccc7c6n(c35)CCC7)C(=O)NC4=O)c2c1')
    [{3, 4, 5, 6, 7, 8, 9}, {1, 2}, {0}]
    """

    if isinstance(cmpd, str):
        cmpd = Chem.MolFromSmiles(cmpd)

    fused_rings = []
    ring_bonds = cmpd.GetRingInfo().BondRings()
    n_rings = len(ring_bonds)

    # List of unassigned rings:
    unassigned = list(range(0, n_rings))

    while len(unassigned) > 0:

        # Start with ring i (root):
        i = unassigned[0]
        del unassigned[0]

        fused_rings.append([i])
        next_r = []

        r1 = ring_bonds[i]

        # Get all unassigned rings which have
        # bonds shared with i:
        for j in unassigned:
            r2 = ring_bonds[j]
            # Check for overlap:
            if len(set(r1) & set(r2)) > 0:
                fused_rings[-1].append(j)
                next_r.append(j)

        # Walk through all neighbouring rings and
        # update with additional neighbours if found:
        while len(next_r) > 0:

            # Move i to next ring, so that this
            # is now the root:
            i = next_r[0]
            del next_r[0]

            # If i has already been assigned then skip:
            if i in unassigned:
                del unassigned[unassigned.index(i)]
            else:
                continue

            r1 = ring_bonds[i]

            for j in unassigned:

                r2 = ring_bonds[j]

                if len(set(r1) & set(r2)) > 0:
                    fused_rings[-1].append(j)
                    next_r.append(j)

        # Need to make final list a set in case some
        # rings are repeated, which can happen if
        # three rings join at the same atom,
        # (see example 4 in docstring)
        fused_rings[-1] = set(fused_rings[-1])

    # Order by number of fused rings in each group:
    fused_rings.sort(key=len, reverse=True)

    # If no rings, return list with an empty set:
    if len(fused_rings) == 0:
        fused_rings.append({})

    return fused_rings


# List of extra descriptors for importing into other modules:
extra_rdkit_descs = [('max_ring_size', max_ring_size), 
                     ('TotalNumProtons', GetTotalNumProtons), 
                     ('TotalNumElectrons', GetTotalNumElectrons), 
                     ('NumHs', GetNumHs), 
                     ('SAscore', sascorer.calculateScore), 
                     ('num_groups_fused_rings', lambda m: len(GetFusedRings(m))), 
                     ('max_fused_rings', lambda m: len(GetFusedRings(m)[0]))]
