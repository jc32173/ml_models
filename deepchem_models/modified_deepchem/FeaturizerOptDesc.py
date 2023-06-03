# flake8: noqa

import numpy as np
import deepchem as dc
from deepchem.feat.base_classes import MolecularFeaturizer
from deepchem.feat.complex_featurizers import ComplexNeighborListFragmentAtomicCoordinates
from deepchem.feat.mol_graphs import ConvMol, WeaveMol
from deepchem.data import DiskDataset
import logging
from typing import Optional, List
from deepchem.utils.typing import RDKitMol, RDKitAtom

from deepchem.feat.graph_features import one_of_k_encoding, one_of_k_encoding_unk, GraphConvConstants, ConvMolFeaturizer

from rdkit import Chem


def atom_features(atom,
                  explicit_H=False,
                  use_chirality=False,
                  features=['atom_type', 
                            'bond_degree', 
                            'implicit_valence', 
                            'formal_charge', 
                            'radical_electrons', 
                            'hybridisation', 
                            'aromatic', 
                            'explicit_H', 
                            'chirality']):
  """Helper method used to compute per-atom feature vectors.

  Many different featurization methods compute per-atom features such as ConvMolFeaturizer, WeaveFeaturizer. This method computes such features.

  Parameters
  ----------
  bool_id_feat: bool, optional
    Return an array of unique identifiers corresponding to atom type.
  explicit_H: bool, optional
    If true, model hydrogens explicitly
  use_chirality: bool, optional
    If true, use chirality information.

  Returns
  -------
  np.ndarray of per-atom features.
  """
  results = []
  if 'atom_type' in features:
    results += one_of_k_encoding_unk(
      atom.GetSymbol(),
      [
        'C',
        'N',
        'O',
        'S',
        'F',
        'Si',
        'P',
        'Cl',
        'Br',
        'Mg',
        'Na',
        'Ca',
        'Fe',
        'As',
        'Al',
        'I',
        'B',
        'V',
        'K',
        'Tl',
        'Yb',
        'Sb',
        'Sn',
        'Ag',
        'Pd',
        'Co',
        'Se',
        'Ti',
        'Zn',
        'H',  # H?
        'Li',
        'Ge',
        'Cu',
        'Au',
        'Ni',
        'Cd',
        'In',
        'Mn',
        'Zr',
        'Cr',
        'Pt',
        'Hg',
        'Pb',
        'Unknown'
      ])
  if 'bond_degree' in features:
    results += one_of_k_encoding(atom.GetDegree(),
                             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
  if 'implicit_valence' in features:
    results += one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6])
  if 'formal_charge' in features:
    results += [atom.GetFormalCharge()]
  if 'radical_electrons' in features:
    results += [atom.GetNumRadicalElectrons()]
  if 'hybridisation' in features:
    results += one_of_k_encoding_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2
              ])
  if 'aromatic' in features:
    results += [atom.GetIsAromatic()]
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
  if not explicit_H and ('explicit_H' in features):
      results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                [0, 1, 2, 3, 4])
  if use_chirality and ('chirality' in features):
      try:
        results += one_of_k_encoding_unk(
            atom.GetProp('_CIPCode'),
            ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
      except:
        results += [False, False
                            ] + [atom.HasProp('_ChiralityPossible')]
  return np.array(results)


class ConvMolFeaturizer_OptDesc(ConvMolFeaturizer):
  """This class implements the featurization to implement Duvenaud graph convolutions.

  Duvenaud graph convolutions [1]_ construct a vector of descriptors for each
  atom in a molecule. The featurizer computes that vector of local descriptors.

  References
  ---------

  .. [1] Duvenaud, David K., et al. "Convolutional networks on graphs for
         learning molecular fingerprints." Advances in neural information
         processing systems. 2015.

  Note
  ----
  This class requires RDKit to be installed.
  """
  name = ['conv_mol']

  def __init__(self, master_atom=False, use_chirality=False,
               atom_properties=[], 
               atom_features=['atom_type', 
                              'bond_degree', 
                              'implicit_valence', 
                              'formal_charge', 
                              'radical_electrons', 
                              'hybridisation', 
                              'aromatic', 
                              'explicit_H', 
                              'chirality']):
    super(ConvMolFeaturizer_OptDesc, self).__init__(
            master_atom=master_atom, use_chirality=use_chirality,
            atom_properties=atom_properties)
    self.atom_features = atom_features

  def _featurize(self, mol):
    """Encodes mol as a ConvMol object."""
    # Get the node features
    idx_nodes = [(a.GetIdx(),
                  np.concatenate((atom_features(
                      a, use_chirality=self.use_chirality, 
                      features=self.atom_features),
                                  self._get_atom_properties(a))))
                 for a in mol.GetAtoms()]

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

    return ConvMol(nodes, canon_adj_list)

  def feature_length(self):
    #return 75 + len(self.atom_properties)
    raise ValueError('Needs fixing in deepchem source code, if ever called (75 only valid if no chirality and no explitic_H)')


#class WeaveFeaturizer(MolecularFeaturizer):
#  """This class implements the featurization to implement Weave convolutions.
#  
#  Weave convolutions were introduced in [1]_. Unlike Duvenaud graph
#  convolutions, weave convolutions require a quadratic matrix of interaction
#  descriptors for each pair of atoms. These extra descriptors may provide for
#  additional descriptive power but at the cost of a larger featurized dataset.
#
#
#  Examples
#  --------
#  >>> import deepchem as dc
#  >>> mols = ["C", "CCC"]
#  >>> featurizer = dc.feat.WeaveFeaturizer()
#  >>> X = featurizer.featurize(mols)
#
#  References
#  ----------
#  .. [1] Kearnes, Steven, et al. "Molecular graph convolutions: moving beyond
#         fingerprints." Journal of computer-aided molecular design 30.8 (2016):
#         595-608.
#
#  Note
#  ----
#  This class requires RDKit to be installed.
#  """
#
#  name = ['weave_mol']
#
#  def __init__(self,
#               graph_distance: bool = True,
#               explicit_H: bool = False,
#               use_chirality: bool = False,
#               max_pair_distance: Optional[int] = None):
#    """Initialize this featurizer with set parameters.
#
#    Parameters
#    ----------
#    graph_distance: bool, (default True)
#      If True, use graph distance for distance features. Otherwise, use
#      Euclidean distance. Note that this means that molecules that this
#      featurizer is invoked on must have valid conformer information if this
#      option is set.
#    explicit_H: bool, (default False) 
#      If true, model hydrogens in the molecule.
#    use_chirality: bool, (default False)
#      If true, use chiral information in the featurization
#    max_pair_distance: Optional[int], (default None)
#      This value can be a positive integer or None. This
#      parameter determines the maximum graph distance at which pair
#      features are computed. For example, if `max_pair_distance==2`,
#      then pair features are computed only for atoms at most graph
#      distance 2 apart. If `max_pair_distance` is `None`, all pairs are
#      considered (effectively infinite `max_pair_distance`)
#    """
#    # Distance is either graph distance(True) or Euclidean distance(False,
#    # only support datasets providing Cartesian coordinates)
#    self.graph_distance = graph_distance
#    # Set dtype
#    self.dtype = object
#    # If includes explicit hydrogens
#    self.explicit_H = explicit_H
#    # If uses use_chirality
#    self.use_chirality = use_chirality
#    if isinstance(max_pair_distance, int) and max_pair_distance <= 0:
#      raise ValueError(
#          "max_pair_distance must either be a positive integer or None")
#    self.max_pair_distance = max_pair_distance
#    if self.use_chirality:
#      self.bt_len = int(GraphConvConstants.bond_fdim_base) + len(
#          GraphConvConstants.possible_bond_stereo)
#    else:
#      self.bt_len = int(GraphConvConstants.bond_fdim_base)
#
#  def _featurize(self, mol):
#    """Encodes mol as a WeaveMol object."""
#    # Atom features
#    idx_nodes = [(a.GetIdx(),
#                  atom_features(
#                      a,
#                      explicit_H=self.explicit_H,
#                      use_chirality=self.use_chirality))
#                 for a in mol.GetAtoms()]
#    idx_nodes.sort()  # Sort by ind to ensure same order as rd_kit
#    idx, nodes = list(zip(*idx_nodes))
#
#    # Stack nodes into an array
#    nodes = np.vstack(nodes)
#
#    # Get bond lists
#    bond_features_map = {}
#    for b in mol.GetBonds():
#      bond_features_map[tuple(sorted([b.GetBeginAtomIdx(),
#                                      b.GetEndAtomIdx()]))] = bond_features(
#                                          b, use_chirality=self.use_chirality)
#
#    # Get canonical adjacency list
#    bond_adj_list = [[] for mol_id in range(len(nodes))]
#    for bond in bond_features_map.keys():
#      bond_adj_list[bond[0]].append(bond[1])
#      bond_adj_list[bond[1]].append(bond[0])
#
#    # Calculate pair features
#    pairs, pair_edges = pair_features(
#        mol,
#        bond_features_map,
#        bond_adj_list,
#        bt_len=self.bt_len,
#        graph_distance=self.graph_distance,
#        max_pair_distance=self.max_pair_distance)
#
#    return WeaveMol(nodes, pairs, pair_edges)
