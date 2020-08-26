
import pytest

import numpy as np
from rdkit import Chem

import ss_utils


def test_random_ordered_smiles():
    smiles = 'CC(=O)Nc1ccc(O)cc1'
    rng = np.random.RandomState(50)

    num_trials = 50
    out_same_str = 0
    for _ in range(num_trials):
        r_smiles = ss_utils.random_ordered_smiles(smiles, rng)
        if r_smiles == smiles: out_same_str += 1
        assert (Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True) ==
                Chem.MolToSmiles(Chem.MolFromSmiles(r_smiles), canonical=True)), "Canonical should always match"

    assert float(out_same_str) / num_trials < 0.3, "Would have expected random SMILES to not match the original so frequently"

def test_graph_as_edge_list_to_canon_smiles():
    node_feats = np.array([[1., 0., 0.],
                           [1., 0., 0.],
                           [0., 0., 1.],
                           [0., 1., 0.],
                           [1., 0., 0.],
                           [1., 0., 0.],
                           [1., 0., 0.],
                           [1., 0., 0.],
                           [0., 0., 1.],
                           [1., 0., 0.],
                           [1., 0., 0.]])

    edge_list = np.array([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1],
                           [1, 3],
                           [3, 1],
                           [3, 4],
                           [4, 3],
                           [4, 5],
                           [5, 4],
                           [5, 6],
                           [6, 5],
                           [6, 7],
                           [7, 6],
                           [7, 8],
                           [8, 7],
                           [7, 9],
                           [9, 7],
                           [9, 10],
                           [10, 9],
                           [10, 4],
                           [4, 10]])

    edge_feature_list = np.array([1., 1., 2., 2., 1., 1., 1., 1., 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1., 1., 1.5, 1.5,
                                  1.5, 1.5, 1.5, 1.5])

    class AtomFeaturizer:
        """
        Atom featurizer takes in an element symbol and returns an array representing its
        one-hot encoding.
        """

        def __init__(self, atoms, feature_size=None):
            self.atm2indx = {k: i for i, k in enumerate(atoms)}
            self.indx2atm = {v: k for k, v in self.atm2indx.items()}
            self.feature_size = feature_size if feature_size is not None else len(atoms)

        def __call__(self, atom_symbol):
            out = np.zeros(self.feature_size)
            out[self.atm2indx[atom_symbol]] = 1.
            return out

    out_smi = ss_utils.graph_as_edge_list_to_canon_smiles(node_feats, edge_list, edge_feature_list, AtomFeaturizer(['C', 'N', 'O']))
    assert out_smi == Chem.MolToSmiles(Chem.MolFromSmiles('CC(=O)Nc1ccc(O)cc1'), canonical=True)


