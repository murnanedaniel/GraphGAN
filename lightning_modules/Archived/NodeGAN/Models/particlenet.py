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
            output_activation=None,
            hidden_activation=hparams["gnn_hidden_activation"]
            )
        
        self.topo_encoder = make_mlp(
            hparams["hidden"],
            [hparams["hidden"]]*hparams["nb_node_layer"] + [hparams["topo_size"]],
            layer_norm=hparams["layernorm"],
            output_activation=None,
            hidden_activation=hparams["gnn_hidden_activation"]
            )
        
        # 2. Decoder network
        self.decoder = ParticleNet(hparams)
        
    def forward(self, x, batch):
        
        x = self.node_encoder(x)
        topo = self.topo_encoder(x)
        
        x, edge_index = self.decoder(x, topo, batch)
            
        return x, edge_index

class ParticleNet(nn.Module):

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
        
        self.topo_net = make_mlp(
            hparams["topo_size"] + (concatenation_factor * hparams["hidden"]),
            [hparams["hidden"]] * hparams["nb_node_layer"] + [hparams["topo_size"]],
            output_activation=None,
            hidden_activation=hparams["gnn_hidden_activation"],
            layer_norm=hparams["layernorm"],
        )
        
        self.feature_net = make_mlp(
            (concatenation_factor + 1) * hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_node_layer"],
            output_activation=None,
            hidden_activation=hparams["gnn_hidden_activation"],
            layer_norm=hparams["layernorm"],
        )
        
        self.edge_net = make_mlp(
            2 * hparams["hidden"] + 2 * hparams["topo_size"],
            [hparams["hidden"]] * hparams["nb_edge_layer"],
            output_activation=None,
            hidden_activation=hparams["gnn_hidden_activation"],
            layer_norm=hparams["layernorm"],
        )
        
        self.features_out = make_mlp(
            hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_node_layer"] + [hparams["input_channels"]],
            output_activation=None,
            hidden_activation=hparams["gnn_hidden_activation"],
            layer_norm=hparams["layernorm"],
        )
                
    def forward(self, x, topo, batch):
                
        # Build first iteration of graph
        edge_index = radius_graph(topo, r=self.hparams["radius"], batch=batch, loop=False)
        
        # Now we're ready to repeat the loop...
        for i in range(self.hparams["n_graph_iters"]):

            x, topo = self.message_step(x, topo, edge_index)
            edge_index = radius_graph(topo, r=self.hparams["radius"], batch=batch, loop=False)
            
        return self.features_out(x), edge_index
    
    def message_step(self, x, topo, edge_index):
        
        start, end = edge_index
        
        # Compute new edge features
        edge_inputs = torch.cat([x[start], x[end], topo[start], topo[end]], dim=-1)
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
        topo_inputs = torch.cat([topo, edge_messages], dim=-1)
        
        x_out = self.feature_net(x_inputs)
        topo_out = self.topo_net(topo_inputs)
        
        x_out += x
        topo_out = F.normalize(topo + topo_out)
         
        return x_out, topo_out
    
        
