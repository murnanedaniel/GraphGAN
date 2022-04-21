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
        
        self.generator_hparams = {}
        self.discriminator_hparams = {}
                
        if ("generator" in hparams.keys()) and ("discriminator" in hparams.keys()):
            self.generator_hparams = self.hparams["generator_hparams"]
            self.discriminator_hparams = self.hparams["discriminator_hparams"]
        else:
            for k, v in self.hparams.items():
                if k.startswith("g_"):
                    self.generator_hparams[k[2:]] = v
                if k.startswith("d_"):
                    self.discriminator_hparams[k[2:]] = v
            
        self.generator = GeneratorGNN(self.generator_hparams)
        self.discriminator = DiscriminatorGNN(self.discriminator_hparams)

    def forward(self, x, edge_index, batch):

        return self.generator(x, edge_index, batch)
        
