# Taken from most recent deepchem version on github (- 2.6.0?), but since running on 2.4.0 some parts (specifically per_atom_fragmentation) have to be commented out and will have to be reinstated when deepchem is upgraded.


import numpy as np
import deepchem as dc
from deepchem.feat.graph_features import ConvMolFeaturizer
from deepchem.feat.mol_graphs import ConvMol #, WeaveMol
import logging
from typing import Optional, List, Union, Iterable

class EmptyConvMolFeaturizer(ConvMolFeaturizer):

  def __init__(self,
               master_atom: bool = False,
               use_chirality: bool = False,
               atom_properties: Iterable[str] = [],
               # Option added in later version of deepchem, will have
               # to uncomment if deepchem is upgraded:
               #per_atom_fragmentation: bool = False
              ):
    super(EmptyConvMolFeaturizer, self).__init__(master_atom=master_atom,
                                                 use_chirality=use_chirality,
                                                 atom_properties=atom_properties,
                                                 # Option added in later version of deepchem:
                                                 #per_atom_fragmentation=per_atom_fragmentation
                                                )

  # Override function which gets standard atom graph features:
  def _featurize(self, mol):
    """Encodes mol as a ConvMol object.
    If per_atom_fragmentation is True,
    then for each molecule a list of ConvMolObjects
    will be created"""

    def per_atom(n, a):
      """
      Enumerates fragments resulting from mol object,
      s.t. each fragment = mol with single atom removed (all possible removals are enumerated)
      Goes over nodes, deletes one at a time and updates adjacency list of lists (removes connections to that node)
      Parameters
      ----------
      n: np.array of nodes (number_of_nodes X number_of_features)
      a: list of nested lists of adjacent node pairs
      """
      for i in range(n.shape[0]):
        new_n = np.delete(n, (i), axis=0)
        new_a = []
        for j, node_pair in enumerate(a):
          if i != j:  # don't need this pair, no more connections to deleted node
            tmp_node_pair = []
            for v in node_pair:
              if v < i:
                tmp_node_pair.append(v)
              elif v > i:
                tmp_node_pair.append(
                    v -
                    1)  # renumber node, because of offset after node deletion
            new_a.append(tmp_node_pair)
        yield new_n, new_a

    # Get the node features
    # Only atom_properties, not standard atom features
    #idx_nodes = [(a.GetIdx(),
    #              np.concatenate((atom_features(
    #                  a, use_chirality=self.use_chirality),
    #                              self._get_atom_properties(a))))
    #             for a in mol.GetAtoms()]
    idx_nodes = [(a.GetIdx(),
                  self._get_atom_properties(a))
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

    # Feature added in later version of deepchem, will have to uncomment when upgraded:
    #if not self.per_atom_fragmentation:
    #  return ConvMol(nodes, canon_adj_list)
    #else:
    #  return [ConvMol(n, a) for n, a in per_atom(nodes, canon_adj_list)]

    return ConvMol(nodes, canon_adj_list)
