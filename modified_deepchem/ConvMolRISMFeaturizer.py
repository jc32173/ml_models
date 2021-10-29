# Check atom ordering in ConvMolFeaturizer parent
# Get a better way of inputing g.dsp file names, rather than trying to recreate them from .mol2/.pdb file names.

import numpy as np
import pandas as pd
import deepchem as dc
from deepchem.feat.base_classes import Featurizer
#from deepchem.feat.base_classes import MolecularFeaturizer
#from deepchem.feat import ConvMolFeaturizer
import logging
from typing import Optional, List, Union, Iterable
from deepchem.utils.typing import RDKitMol, RDKitAtom
from deepchem.feat.mol_graphs import ConvMol

from rdkit import Chem
from rdkit.Chem import rdmolfiles
from rdkit.Chem import rdmolops
from rdkit.Chem.rdchem import Mol
import os

logger = logging.getLogger(__name__)

# No point inheriting from ConvMolFeaturizer or MolecularFeaturizer as 
# all parts of those classes are overwritten:
#class ConvMolRISMFeaturizer(ConvMolFeaturizer):
#class ConvMolRISMFeaturizer(MolecularFeaturizer):
class ConvMolRISMFeaturizer(Featurizer):
#class ConvMolRISMFeaturizer(UserDefinedFeaturizer):
    """Class to add RISM correlation functions to graph nodes 
    for GraphConv NN.
    
    Class is based heavily on ConvMolFeaturizer:
    https://github.com/deepchem/deepchem/blob/9de3977833b1834e1672d3f4f1ecc4345cef9b81/deepchem/feat/graph_features.py#L627
    
    Note
    ----
    This class requires RDKit to be installed.
    """

#     # If inheriting from ConvMol:
#     def __init__(self,
#                  step: int = 1,
#                  *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         # Keep RISM data every n steps:
#         self.rism_step = step
    
    # Not inheriting from ConvMolFeaturizer, but may eventually want to 
    # add some of these attributes, e.g.
    #    per_atom_fragmentation - required for DAG?
    #    atom_properties - Keep atoms properties alongside RISM data?
    def __init__(self,
                 start: int = 0,
                 stop: int = -1,
                 step: int = 1,
                 #points: int = 6000,
                 pdb_file_idx=0,
                 mol2_file_idx=None,
                 gdsp_file_idx=-1,
                 per_atom_fragmentation=False,
                 master_atom=None):
        self.rism_start = start
        self.rism_stop = stop
        self.rism_step = step
        self.pdb_file_idx = pdb_file_idx
        self.mol2_file_idx = mol2_file_idx
        self.gdsp_file_idx = gdsp_file_idx
        self.master_atom = master_atom
        self.per_atom_fragmentation = per_atom_fragmentation
        #self.rism_points = points
        # From ConvMolFeaturizer:
        self.dtype = object
        
    # Function to replace atom_features (not part of 
    # ConvMolFeaturizer class, but defined in the same module:
    # https://github.com/deepchem/deepchem/blob/9de3977833b1834e1672d3f4f1ecc4345cef9b81/deepchem/feat/graph_features.py#L274)
    def rism_atom_features(self,
                           atom,
                           gdsp,
                           start=0,
                           stop=-1,
                           step=1):
        """
        """
        # RDKit is loaded in featurizer, so not needed here:
        #from rdkit import Chem
        
        # NEED TO CHECK THAT ATOM IDXS LINE UP:
        atom_idx = atom.GetIdx()

        # See RISM user manual
        # Oxygen atom of RDF:
        gdsp_O = gdsp[start:stop:step,2*atom_idx + 1]
        #gdsp_H = gdsp[start:stop:step,2*atom_idx + 2]

        return gdsp_O

    # Overload a couple of methods:
    # Copied from MolecularFeaturizer with only change being to add gdsp
    # to call to self._featurize:
    def featurize(self,
                  #molecules: Union[RDKitMol, str, Iterable[RDKitMol], Iterable[str]],
                  mol_files: Union[str, Iterable[str]],
                  #gdsp_files: Union[str, Iterable[str]],
                  log_every_n: int = 1000) -> np.ndarray:
        """
        Parameters
        ----------
        mol_struct_files: .mol2 or .pdb filename string or iterable sequence of filenames
        gdsp_files: g.dsp filename string or iterable sequence of filenames
          (Must be in same order as mol_struct_files.)
        log_every_n: int, default 1000
          Logging messages reported every `log_every_n` samples.

        Returns
        -------
        features: np.ndarray
          A numpy array containing a featurized representation of `datapoints`.
        """
    
    # In ConvMolFeaturizer, the parent featurize method is called first and 
    # then additional code to deal with per_atom_fragmentation is added.  However, 
    # since the parent featurize method will call the _featurize method of this
    # class, and this now takes the g_dsp data as an additional parameter, all of
    # the featurize function from the parent has been copied here, rather than
    # using super().  Have also added checks that gdsp_files is same length as 
    # molecules:

    #features = super(ConvMolFeaturizer, self).featurize(
    #    molecules, log_every_n=1000)

    # Based on original featurize method code from MolecularFeaturizer:
        try:
            from rdkit import Chem
            from rdkit.Chem import rdmolfiles
            from rdkit.Chem import rdmolops
            from rdkit.Chem.rdchem import Mol
            import os
        except ModuleNotFoundError:
            raise ImportError("This class requires RDKit to be installed.")

        # MAYBE BETTER TO JUST LOAD MOL FROM mol2 FILE?

        # Special case handling of single molecule
        #if isinstance(molecules, str) or isinstance(molecules, Mol):
        #    molecules = [molecules]
        #else:
        #    # Convert iterables to list
        #    molecules = list(molecules)

        # Have to use eval to convert string to tuple:
        if isinstance(mol_files, str):
            mol_files = [eval(mol_files)]
        else:
            # Convert iterables to list
            mol_files = [eval(mf) for mf in mol_files]

        # g.dsp files:
        # BETTER TO GET THESE AS INPUTS:
        #mol_file_ext = mol_struct_files[0].split('.')[-1]
        #if mol_file_ext == 'mol2':
        #    gdsp_files = [os.path.dirname(mf)+'/g.dsp' for mf in mol_struct_files]
        #elif mol_file_ext == 'pdb':
        #    gdsp_files = ['/users/xpb20111/AqSolDB/RISM/rism_mol_'+(mf.split('/')[-1]).split('_')[0]+'_conf0/g.dsp' for mf in mol_struct_files]

        # Use pdb file if pdb and mol2 files given:
        if self.pdb_file_idx is not None:
            mol_struct_files = [mf[self.pdb_file_idx] for mf in mol_files]
            # Check file extension?
        elif self.mol2_file_idx is not None:
            mol_struct_files = [mf[self.mol2_file_idx] for mf in mol_files]

        gdsp_files = [mf[self.gdsp_file_idx] for mf in mol_files]

        # and single g.dsp file:
        #if isinstance(gdsp_files, str):
        #    gdsp_files = [gdsp_files]
        #else:
        #    # Convert iterables to list
        #    gdsp_files = list(gdsp_files)

        # Check same number of molecules as g.dsp files:
        if len(mol_struct_files) != len(gdsp_files):
            raise ValueError("Number of .mol2/.pdb and g.dsp files must be the same.")

        features = []
        for i, mol_file in enumerate(mol_struct_files):
            if i % log_every_n == 0:
                logger.info("Featurizing datapoint %i" % i)

            try:
                #if isinstance(mol, str):
                #    # mol must be a RDKit Mol object, so parse a SMILES
                #    mol = Chem.MolFromSmiles(mol)
                #mol = Chem.MolFromMol2File(mol, removeHs=False)
                #    # SMILES is unique, so set a canonical order of atoms
                #    # NEED TO THINK ABOUT THIS REORDING:
                #    new_order = rdmolfiles.CanonicalRankAtoms(mol)
                #    mol = rdmolops.RenumberAtoms(mol, new_order)

                features.append(self._featurize(mol_file, gdsp_files[i]))

            except Exception as e:
                #if isinstance(mol, Chem.rdchem.Mol):
                #    mol = Chem.MolToSmiles(mol)
                logger.warning(
                    "Failed to featurize datapoint %d, %s. Appending empty array", i,
                    mol_file)
                logger.warning("Exception message: {}".format(e))
                features.append(np.array([]))

        # End of code from parent class, but don't return here in case adding additional code
        features = np.asarray(features)
    
#    The additional code for per_atom_fragmentation could be added in here:
#     if self.per_atom_fragmentation:
#         # create temporary valid ids serving to filter out failed featurizations from every sublist
#         # of features (i.e. every molecules' frags list), and also totally failed sublists.
#         # This makes output digestable by Loaders
#         valid_frag_inds = [[
#             True if np.array(elt).size > 0 else False for elt in f
#         ] for f in features]
#         features = [[elt
#                      for (is_valid, elt) in zip(l, m)
#                      if is_valid]
#                   for (l, m) in zip(valid_frag_inds, features)
#                   if any(l)]

        return features

    def _featurize(self, mol_file, g_dsp_file):
        """Encodes mol as a ConvMol object.
        If per_atom_fragmentation is True,
        then for each molecule a list of ConvMolObjects
        will be created"""

#     # Additional code for per_atom_fragmentation could be added in here:
#     # (see ConMolFeaturizer)
#         def per_atom(n, a):
#             """
#             Enumerates fragments resulting from mol object,
#             s.t. each fragment = mol with single atom removed (all possible removals are enumerated)
#             Goes over nodes, deletes one at a time and updates adjacency list of lists (removes connections to that node)
#             Parameters
#             ----------
#             n: np.array of nodes (number_of_nodes X number_of_features)
#             a: list of nested lists of adjacent node pairs
#             """
#             for i in range(n.shape[0]):
#                 new_n = np.delete(n, (i), axis=0)
#                 new_a = []
#                 for j, node_pair in enumerate(a):
#                     if i != j:  # don't need this pair, no more connections to deleted node
#                         tmp_node_pair = []
#                     for v in node_pair:
#               if v < i:
#                 tmp_node_pair.append(v)
#               elif v > i:
#                 tmp_node_pair.append(
#                     v -
#                     1)  # renumber node, because of offset after node deletion
#             new_a.append(tmp_node_pair)
#         yield new_n, new_a

        # Get mol and atom order from mol2 or pdb file:
        # https://www.rdkit.org/docs/source/rdkit.Chem.rdmolfiles.html
        # MolFromMol2File creates molecule with atom indices which reflect the order of atoms
        # in the mol2/pdb file.

        mol_file_ext = mol_file.split('.')[-1]
        #print(mol_file_ext)
        if mol_file_ext == 'mol2':
            mol = Chem.MolFromMol2File(mol_file,
                                       removeHs=False)
        # May have to make sanitize and/or cleanupSubstructures False as some errors when getting 
        # molecules from mol2 files, may be better to get mol from SMILES instead.
        #                           sanitize=False,
        #                           cleanupSubstructures=False)

        elif mol_file_ext == 'pdb':
            mol = Chem.MolFromPDBFile(mol_file,
                                      removeHs=False)

        #if mol is None:
        #    #Deal
        canon_order = rdmolfiles.CanonicalRankAtoms(mol)

        # Load g.dsp file for specifc molecule:
        g_dsp = pd.read_csv(g_dsp_file,
                            sep='\s+',
                            engine='python',
                            header=None).to_numpy()

        # Get the node features
        idx_nodes = [(canon_order[a.GetIdx()],
                      self.rism_atom_features(a, g_dsp, self.rism_start, self.rism_stop, self.rism_step))
                      for a in mol.GetAtoms()]

        # Reorder mol to canonical order:
        mol = rdmolops.RenumberAtoms(mol, canon_order)

        idx_nodes.sort()  # Sort by ind to ensure same order as rd_kit
        idx, nodes = list(zip(*idx_nodes))

        # Stack nodes into an array
        nodes = np.vstack(nodes)
        if self.master_atom:
            master_atom_features = np.expand_dims(np.mean(nodes, axis=0), axis=0)
            nodes = np.concatenate([nodes, master_atom_features], axis=0)

        # Get bond lists with reverse edges included
        edge_list = [
            (b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()
        ]
        # Get canonical adjacency list
        canon_adj_list = [[] for mol_id in range(len(nodes))]
        for edge in edge_list:
            canon_adj_list[edge[0]].append(edge[1])
            canon_adj_list[edge[1]].append(edge[0])

        if self.master_atom:
            fake_atom_index = len(nodes) - 1
            for index in range(len(nodes) - 1):
                canon_adj_list[index].append(fake_atom_index)

        if not self.per_atom_fragmentation:
            return ConvMol(nodes, canon_adj_list)
        else:
            return [ConvMol(n, a) for n, a in per_atom(nodes, canon_adj_list)]

    def feature_length(self):
        return self.rism_stop - self.rism_start // self.rism_step

    # Remaining functions from ConvMolFeaturizer: __hash__, __eq__ and _get_atom_properties not used
