"""
This module contains some helper functions for our main notebook
"""

import collections
import itertools
import time
import typing
from dataclasses import dataclass

import numpy as np
import altair as alt
import pandas as pd
import matplotlib.pyplot as plt


from rdkit import Chem
from rdkit.Chem import AllChem

import torch
from torch.utils import data
from torch import nn
from torch import optim
from torch.nn import functional as F
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage

from ignite.contrib.handlers import ProgressBar


@dataclass
class TrainParams:
    batch_size: int = 64
    val_batch_size: int = 64
    learning_rate: float = 1e-3
    num_epochs: int = 10
    device: typing.Optional[str] = 'cpu'  # <-- I have not run this on the GPU yet, so that may need some debugging.


class SmilesRegressionDataset(data.Dataset):
    """
    Dataset that holds SMILES molecule data along with an associated single
    regression target.
    """

    def __init__(self, smiles_list: typing.List[str],
                 regression_target_list: typing.List[float],
                 transform: typing.Optional[typing.Callable] = None):
        """
        :param smiles_list: list of SMILES strings represnting the molecules
        we are regressing on.
        :param regression_target_list: list of targets
        :param transform: an optional transform which will be applied to the
        SMILES string before it is returned.
        """
        self.smiles_list = smiles_list
        self.regression_target_list = regression_target_list
        self.transform = transform

        assert len(self.smiles_list) == len(self.regression_target_list), \
            "Dataset and targets should be the same length!"

    def __getitem__(self, index):
        x, y = self.smiles_list[index], self.regression_target_list[index]
        if self.transform is not None:
            x = self.transform(x)
        y = torch.tensor([y], dtype=torch.float32)
        return x, y

    def __len__(self):
        return len(self.smiles_list)

    @classmethod
    def create_from_df(cls, df: pd.DataFrame, smiles_column: str = 'smiles',
                       regression_column: str = 'y', transform=None):
        """
        convenience method that takes in a Pandas dataframe and turns it
        into an   instance of this class.
        :param df: Dataframe containing the data.
        :param smiles_column: name of column that contains the x data
        :param regression_column: name of the column which contains the
        y data (i.e. targets)
        :param transform: a transform to pass to class's constructor
        """
        smiles_list = [x.strip() for x in df[smiles_column].tolist()]
        targets = [float(y) for y in df[regression_column].tolist()]
        return cls(smiles_list, targets, transform)


def train_neural_network(train_df: pd.DataFrame, val_df: pd.DataFrame,
                          smiles_col:str, regression_column:str,
                         transform: typing.Callable,
                         neural_network: nn.Module,
                         params: typing.Optional[TrainParams]=None,
                         collate_func: typing.Optional[typing.Callable]=None):
    """
    Trains a PyTorch NN module on train dataset, validates it each epoch and returns a series of useful metrics
    for further analysis. Note the networks parameters will be changed in place.

    :param train_df: data to use for training.
    :param val_df: data to use for validation.
    :param smiles_col: column name for SMILES data in Dataframe
    :param regression_column: column name for the data we want to regress to.
    :param transform: the transform to apply to the datasets to create new ones suitable for working with neural network
    :param neural_network: the PyTorch nn.Module to train
    :param params: the training params eg number of epochs etc.
    :param collate_func: collate_fn to pass to dataloader constructor. Leave as None to use default.
    """
    if params is None:
        params = TrainParams()

    # Update the train and valid datasets with new parameters
    train_dataset = SmilesRegressionDataset.create_from_df(train_df, smiles_col, regression_column, transform=transform)
    val_dataset = SmilesRegressionDataset.create_from_df(val_df, smiles_col, regression_column, transform=transform)
    print(f"Train dataset is of size {len(train_dataset)} and valid of size {len(val_dataset)}")

    # Put into dataloaders
    train_dataloader = data.DataLoader(train_dataset, params.batch_size, shuffle=True,
                                       collate_fn=collate_func, num_workers=1)
    val_dataloader = data.DataLoader(val_dataset, params.val_batch_size, shuffle=False, collate_fn=collate_func,
                                       num_workers=1)

    # Optimizer
    optimizer = optim.Adam(neural_network.parameters(), lr=params.learning_rate)

    # Work out what device we're going to run on (ie CPU or GPU)
    device = params.device

    # We're going to use PyTorch Ignite to take care of the majority of the training boilerplate for us
    # see https://pytorch.org/ignite/
    # in particular we are going to follow the example
    # https://github.com/pytorch/ignite/blob/53190db227f6dda8980d77fa5351fa3ddcdec6fb/examples/contrib/mnist/mnist_with_tqdm_logger.py
    def prepare_batch(batch, device, non_blocking):
        x, y = batch
        return x.to(device), y.to(device)

    trainer = create_supervised_trainer(neural_network, optimizer, F.mse_loss, device=device, prepare_batch=prepare_batch)
    evaluator = create_supervised_evaluator(neural_network,
                                            metrics={'loss': Loss(F.mse_loss)},
                                            device=device, prepare_batch=prepare_batch)
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names='all')

    train_loss_list = []
    val_lost_list = []
    val_times_list = []

    @trainer.on(Events.EPOCH_COMPLETED | Events.STARTED)
    def log_training_results(engine):
        evaluator.run(train_dataloader)
        metrics = evaluator.state.metrics
        loss = metrics['loss']
        pbar.log_message("Epoch - {}".format(engine.state.epoch))
        pbar.log_message(
            "Training Results - Epoch: {}  Avg loss: {:.2f}"
                .format(engine.state.epoch, loss)
        )
        train_loss_list.append(loss)

    @trainer.on(Events.EPOCH_COMPLETED | Events.STARTED)
    def log_validation_results(engine):
        s_time = time.time()
        evaluator.run(val_dataloader)
        e_time = time.time()
        metrics = evaluator.state.metrics
        loss = metrics['loss']
        pbar.log_message(
            "Validation Results - Epoch: {} Avg loss: {:.2f}"
                .format(engine.state.epoch, loss))

        pbar.n = pbar.last_print_n = 0
        val_lost_list.append(loss)
        val_times_list.append(e_time - s_time)

    # We can now train our network!
    trainer.run(train_dataloader, max_epochs=params.num_epochs)

    # Having trained it wee are now also going to run through the validation set one
    # last time to get the actual predictions
    val_predictions = []
    neural_network.eval()
    for batch in val_dataloader:
        x, y = batch
        x = x.to(device)
        y_pred = neural_network(x)
        assert (y_pred.shape) == (y.shape), "If this is not true then would cause problems in loss"
        val_predictions.append(y_pred.cpu().detach().numpy())
    neural_network.train()
    val_predictions = np.concatenate(val_predictions)

    # Create a table of useful metrics (as part of the information we return)
    total_number_params = sum([v.numel() for v in  neural_network.parameters()])
    out_table = [
        ["Num params", f"{total_number_params:.2e}"],
        ["Minimum train loss", f"{np.min(train_loss_list):.3f}"],
        ["Mean validation time", f"{np.mean(val_times_list):.3f}"],
        ["Minimum validation loss", f"{np.min(val_lost_list):.3f}"]
     ]

    # We will create a dictionary of results.
    results = dict(
        train_loss_list=train_loss_list,
        val_lost_list=val_lost_list,
        val_times_list=val_times_list,
        out_table=out_table,
        val_predictions=val_predictions
    )
    return results


def plot_train_and_val_using_mpl(train_loss, val_loss):
    """
    Plots the train and validation loss using Matplotlib
    """
    assert len(train_loss) == len(val_loss)

    f, ax = plt.subplots()
    x = np.arange(len(train_loss))
    ax.plot(x, np.array(train_loss), label='train')
    ax.plot(x, np.array(val_loss), label='val')
    ax.legend()
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    return f, ax


def plot_train_and_val_using_altair(train_loss, val_loss):
    """
    Plots the train and validation loss using Altair -- see https://altair-viz.github.io/gallery/multiline_tooltip.html
    This should result in an interactive plot which we can mouseover.
    """
    assert len(train_loss) == len(val_loss)
    source = pd.DataFrame.from_dict({"train_loss": train_loss, "val_loss": val_loss, 'epoch': np.arange(len(train_loss))})
    source = source.melt('epoch', var_name='category', value_name='loss')

    # Create a selection that chooses the nearest point & selects based on x-value
    nearest = alt.selection(type='single', nearest=True, on='mouseover',
                            fields=['epoch'], empty='none')

    # The basic line
    line = alt.Chart(source).mark_line(interpolate='basis').encode(
        x='epoch:Q',
        y='loss:Q',
        color='category:N'
    )

    # Transparent selectors across the chart. This is what tells us
    # the x-value of the cursor
    selectors = alt.Chart(source).mark_point().encode(
        x='epoch:Q',
        opacity=alt.value(0),
    ).add_selection(
        nearest
    )

    # Draw points on the line, and highlight based on selection
    points = line.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )

    # Draw text labels near the points, and highlight based on selection
    text = line.mark_text(align='left', dx=5, dy=-5).encode(
        text=alt.condition(nearest, 'loss:Q', alt.value(' '))
    )

    # Draw a rule at the location of the selection
    rules = alt.Chart(source).mark_rule(color='gray').encode(
        x='epoch:Q',
    ).transform_filter(
        nearest
    )

    # Put the five layers into a chart and bind the data
    return alt.layer(
        line, selectors, points, rules, text
    ).properties(
        width=600, height=300
    )


def random_ordered_smiles(smiles: str, rng: np.random.RandomState) -> str:
    """
    Returns a randomly chosen SMILES string that represents a molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    new_perm = rng.permutation(mol.GetNumAtoms()).tolist()
    new_mol = Chem.RenumberAtoms(mol,new_perm)
    return Chem.MolToSmiles(new_mol, canonical=False)


def graph_as_edge_list_to_canon_smiles(node_features: np.ndarray, edge_list: np.ndarray, edge_features: np.ndarray,
                                 atom_featurizer) -> str:
    """
    Converts graph as node_features/edge_list back to a canonical SMILES representation -- note that not always
    reversible.

    Convenience method to check that one of the notebook tasks has been done correctly -- see notebook for further
    details.
    """
    # 1. Basic checks
    if edge_list.shape[0] != edge_features.shape[0]:
        raise RuntimeError("edge_list and edge_features should have the same first dimension")

    if node_features.shape[1] != len(atom_featurizer.indx2atm):
        raise RuntimeError("node_features should be consistent with dimensionality output by atom_featurizer")

    # 2. Add atoms to molecule
    mol = Chem.RWMol()
    for feat in node_features:
        symb = atom_featurizer.indx2atm[int(np.argmax(feat))]
        mol.AddAtom(Chem.Atom(symb))

    # 3. Add bonds and check that each is referred to twice (one in each direction)
    bonds_to_count = collections.defaultdict(lambda: 0)
    # ^ we'll use this to check that each bond is referred to in edge_list twice -- for both dirs.
    bnd_double_to_type = {
        1: AllChem.BondType.SINGLE,
        2: AllChem.BondType.DOUBLE,
        1.5: AllChem.BondType.AROMATIC,
        3: AllChem.BondType.TRIPLE
    }
    for bnd, bnd_type in zip(edge_list, edge_features):
        invariant_bnd_repr = frozenset(bnd.tolist())
        if invariant_bnd_repr not in bonds_to_count:
            mol.AddBond(*map(int, bnd), bnd_double_to_type[float(bnd_type)])
        bonds_to_count[invariant_bnd_repr] += 1

    if not all([v == 2 for v in bonds_to_count.values()]):
        raise RuntimeError("Did you include the bonds going both ways...?")

    # 4. Turn into canonical SMILES and we're done!
    smiles = Chem.MolToSmiles(mol, canonical=True)
    return smiles
