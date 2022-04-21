import sys

import torch.nn as nn
from torch.nn import Linear
import torch
from torch_scatter import scatter_add, scatter_mean, scatter_max
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import global_add_pool

from ..gnn_base import GNNBase
from .particlenet import GeneratorGNN
from .discriminator import DiscriminatorGNN
from ..utils import make_mlp

class GanGNN(GNNBase):

    """
    An interaction network class
    """

    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """
        self.generator = GeneratorGNN(hparams["generator_hparams"])
        self.discriminator = DiscriminatorGNN(hparams["generator_hparams"])

    def forward(self, x, batch):

        return self.generator(x, batch)
        
