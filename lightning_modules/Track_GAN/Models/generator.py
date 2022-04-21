import sys

import torch.nn as nn
from torch.nn import Linear
import torch
from torch_scatter import scatter_add, scatter_mean, scatter_max
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import global_add_pool
from torch_cluster import radius_graph
import torch.nn.functional as F


from ..gnn_base import GNNBase
from ..utils import make_mlp

class GeneratorGNN(nn.Module):
    
    def __init__(self, hparams):
        
        super().__init__()
        
        self.node_encoder = make_mlp(
            hparams["input_channels"],
            [hparams["hidden"]]* hparams["nb_node_layer"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=None,
            hidden_activation=hparams["gnn_hidden_activation"]
            )
        
        self.edge_encoder = make_mlp(
            2*hparams["hidden"],
            [hparams["hidden"]]* hparams["nb_edge_layer"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=None,
            hidden_activation=hparams["gnn_hidden_activation"]
            )
        
        # 2. Decoder network
        self.decoder = InteractionNet(hparams)
        
    def forward(self, x, edge_index, batch):
        
        x = self.node_encoder(x)
        e = self.edge_encoder(torch.cat([x[edge_index[0]], x[edge_index[1]]], axis=-1))
        
        x, edge_score = self.decoder(x, e, edge_index, batch)
            
        return x, edge_score

class InteractionNet(nn.Module):

    """
    An interaction network class
    """

    def __init__(self, hparams):
        super().__init__()
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """
        self.hparams = hparams
        concatenation_factor = 2 if (self.hparams["aggregation"] in ["sum_max", "mean_max"]) else 1
        
        
        self.feature_net = make_mlp(
            (concatenation_factor + 1) * hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_node_layer"],
            output_activation=None,
            hidden_activation=hparams["gnn_hidden_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
        )
        
        self.edge_net = make_mlp(
            3 * hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_edge_layer"],
            output_activation=None,
            hidden_activation=hparams["gnn_hidden_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
        )
        
        self.features_out = make_mlp(
            hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_node_layer"] + [hparams["input_channels"]],
            output_activation=None,
            hidden_activation=hparams["gnn_hidden_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
        )
        
        self.scores_out = make_mlp(
            3 * hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_edge_layer"] + [1],
            output_activation=None,
            hidden_activation=hparams["gnn_hidden_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
        )
                
    def forward(self, x, e, edge_index, batch):
            
        start, end = edge_index
        
        # Now we're ready to repeat the loop...
        for i in range(self.hparams["n_graph_iters"]):

            x, e = self.message_step(x, e, edge_index)
            
        edge_score = self.scores_out(torch.cat([x[start], x[end], e], dim=-1))
            
        return self.features_out(x), torch.sigmoid(edge_score)
    
    def message_step(self, x, e, edge_index):
        
        start, end = edge_index
        
        # Compute new edge features
        edge_inputs = torch.cat([x[start], x[end], e], dim=-1)
        edge_features = self.edge_net(edge_inputs)  
        
        # Compute new node features        
        if self.hparams["aggregation"] == "sum":  
            edge_messages = scatter_add(edge_features, end, dim=0, dim_size=x.shape[0]) 
        
        elif self.hparams["aggregation"] == "max":
            edge_messages = scatter_max(edge_features, end, dim=0, dim_size=x.shape[0])[0]

        elif self.hparams["aggregation"] == "sum_max":
            edge_messages = torch.cat([scatter_max(edge_features, end, dim=0, dim_size=x.shape[0])[0],
                                 scatter_add(edge_features, end, dim=0, dim_size=x.shape[0])], dim=-1)
        elif self.hparams["aggregation"] == "mean_max":
            edge_messages = torch.cat([scatter_max(edge_features, end, dim=0, dim_size=x.shape[0])[0],
                                 scatter_mean(edge_features, end, dim=0, dim_size=x.shape[0])], dim=-1)
        
        x_inputs = torch.cat([x, edge_messages], dim=-1)
        x_out = self.feature_net(x_inputs)
        x_out = x + x_out
        e_out = e + edge_features
         
        return x_out, e_out
    
        
