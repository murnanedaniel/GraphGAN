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
        
        print(hparams)
        
        self.node_encoder = make_mlp(
            hparams["input_node_channels"],
            [hparams["hidden"]]* hparams["nb_node_layer"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=hparams["output_hidden_activation"],
            hidden_activation=hparams["gnn_hidden_activation"]
            )
        
        self.edge_encoder = make_mlp(
            2*hparams["hidden"],
            [hparams["hidden"]]* hparams["nb_edge_layer"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=hparams["output_hidden_activation"],
            hidden_activation=hparams["gnn_hidden_activation"]
            )
        
        self.graph_encoder = make_mlp(
            hparams["input_graph_channels"],
            [hparams["hidden"]]* hparams["nb_node_layer"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
            output_activation=hparams["output_hidden_activation"],
            hidden_activation=hparams["gnn_hidden_activation"]
            )
        
        # 2. Decoder network
        self.decoder = InteractionNet(hparams)
        
    def forward(self, x, graph_data, edge_index, batch, repeater):
               
        start, end = edge_index
        
        x = self.node_encoder(torch.cat([x], axis=-1))
        e = self.edge_encoder(torch.cat([x[start], x[end]], axis=-1))
        g = self.graph_encoder(graph_data)
        
        x = self.decoder(x, e, g, edge_index, batch, repeater)
            
        return x

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
        concatenation_factor = 2 if (self.hparams["aggregation"] in ["sum_max", "mean_max", "sum_mean"]) else 1
        
        
        self.feature_net = make_mlp(
            (concatenation_factor + 2) * hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_node_layer"],
            output_activation=hparams["output_hidden_activation"],
            hidden_activation=hparams["gnn_hidden_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
        )
        
        self.graph_net = make_mlp(
            hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_node_layer"],
            output_activation=hparams["output_hidden_activation"],
            hidden_activation=hparams["gnn_hidden_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
        )
        
        self.edge_net = make_mlp(
            3 * hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_edge_layer"],
            output_activation=hparams["output_hidden_activation"],
            hidden_activation=hparams["gnn_hidden_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
        )
        
        self.features_out = make_mlp(
            hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_node_layer"] + [hparams["output_channels"]],
            output_activation=hparams["final_output_activation"],
            hidden_activation=hparams["gnn_hidden_activation"],
            layer_norm=hparams["layernorm"],
            batch_norm=hparams["batchnorm"],
        )
                
    def forward(self, x, e, g, edge_index, batch, repeater):
            
        start, end = edge_index
        
        # Now we're ready to repeat the loop...
        for i in range(self.hparams["n_graph_iters"]):

            x, e, g = self.message_step(x, start, end, e, g, batch, repeater)
                        
        return self.features_out(x)
    
    def message_step(self, x, start, end, e, g, batch, repeater):
        
        # Compute new node features        
        if self.hparams["aggregation"] == "sum":  
            edge_messages = scatter_add(e, end, dim=0, dim_size=x.shape[0]) 
        
        elif self.hparams["aggregation"] == "max":
            edge_messages = scatter_max(e, end, dim=0, dim_size=x.shape[0])[0]

        elif self.hparams["aggregation"] == "sum_max":
            edge_messages = torch.cat([scatter_max(e, end, dim=0, dim_size=x.shape[0])[0],
                                 scatter_add(e, end, dim=0, dim_size=x.shape[0])], dim=-1)
            
        elif self.hparams["aggregation"] == "mean_max":
            edge_messages = torch.cat([scatter_max(e, end, dim=0, dim_size=x.shape[0])[0],
                                 scatter_mean(e, end, dim=0, dim_size=x.shape[0])], dim=-1)
        
        elif self.hparams["aggregation"] == "sum_mean":
            edge_messages = torch.cat([scatter_mean(e, end, dim=0, dim_size=x.shape[0]),
                                 scatter_add(e, end, dim=0, dim_size=x.shape[0])], dim=-1)
        
        # Compute new node features
        x_inputs = torch.cat([x, edge_messages, g], dim=-1)
        # print(x_inputs.shape, x.shape, edge_messages.shape, g.shape)
        x_out = self.feature_net(x_inputs)
        x_out += x
        
        # Compute new graph features
        graph_inputs = global_add_pool(x_out, batch)
        g_out = self.graph_net(graph_inputs).repeat_interleave(repeater, dim=0) 
        g_out += g
        
        # Compute new edge features
        edge_inputs = torch.cat([x_out[start], x_out[end], e], dim=-1)
        e_out = self.edge_net(edge_inputs)  
        e_out += e
        
        return x_out, e_out, g_out
    
        
