import sys

import torch.nn as nn
from torch.nn import Linear
import torch
from torch_scatter import scatter_add, scatter_mean, scatter_max
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import global_add_pool

from ..gnn_base import GNNBase
from ..utils import make_mlp

class GeneratorGNN(nn.Module):
    
    def __init__(self, hparams):
        
        super().__init__()
        
        # Define 3 sampled points:
        # 1) n1 -> MLP -> number of nodes
        # 2) n2 -> MLP -> global node feature
        # 3) n3 -> MLP -> local node topo
        
        # 1. Node builder - takes a [latent_size] point, returns scalar
        self.node_builder = make_mlp(
            hparams["latent_size"],
            [hparams["hidden"]]* hparams["nb_node_layer"] + [1],
            layer_norm=hparams["layernorm"],
            output_activation=hparams["output_hidden_activation"],
            hidden_activation=hparams["gnn_hidden_activation"]
            )
        
        self.node_encoder = make_mlp(
            1 + hparams["latent_size"],
            [hparams["hidden"]]* hparams["nb_node_layer"],
            layer_norm=hparams["layernorm"],
            output_activation=hparams["output_hidden_activation"],
            hidden_activation=hparams["gnn_hidden_activation"]
            )
        
        self.topo_encoder = make_mlp(
            hparams["hidden"] + hparams["latent_size"],
            [hparams["hidden"]]*hparams["nb_node_layer"] + [hparams["topo_size"]],
            layer_norm=hparams["layernorm"],
            output_activation=None,
            hidden_activation=hparams["gnn_hidden_activation"]
            )
        
        # 2. Decoder network
        self.decoder = ParticleNet(hparams)
        
    def forward(self):
        
        scalar_sample = torch.randn(self.hparams["latent_size"]) #Use as_type
        scalar_feature = self.node_builder(sample)
        
        num_nodes = torch.round(scalar_feature)
        global_sample = torch.randn(self.hparams["latent_size"]) #Use as_type
        global_feature = self.node_encoder(torch.cat([scalar_feature, global_sample], dim=-1)
        
        x = global_feature.repeat(num_nodes, 1)      
        
        local_sample = torch.randn(num_nodes, self.hparams["latent_size"])
        local_topo = self.topo_encoder(torch.cat([x, global_feature], dim=-1))
        
        _, x, edge_index = self.decoder(x, local_topo)
               

class ParticleNet(nn.Module):

    """
    An interaction network class
    """

    def __init__(self, hparams):
        super().__init__(hparams)
        """
        Initialise the Lightning Module that can scan over different GNN training regimes
        """
        
        # Takes 
        self.node_encoder = make_mlp(
            hparams["input_channels"],
            [hparams["hidden"]] * hparams["nb_node_layer"],
            output_activation=None,
            hidden_activation=hparams["gnn_hidden_activation"],
            layer_norm=hparams["layernorm"],
        )
        
        self.topo_net = make_mlp(
            hparams["input_channels"],
            [hparams["hidden"]] * hparams["nb_node_layer"],
            output_activation=None,
            hidden_activation=hparams["gnn_hidden_activation"],
            layer_norm=hparams["layernorm"],
        )
        
        self.feature_net
        
        self.edge_net = make_mlp(
            2 * hparams["hidden"],
            [hparams["hidden"]] * hparams["nb_edge_layer"],
            output_activation=None,
            hidden_activation=hparams["gnn_hidden_activation"],
            layer_norm=hparams["layernorm"],
        )
        
        self.global_output = 
        
    def forward(self, x, topo):
        
        edge_index = build_edges(topo, self.hparams["radius"]) # Add build edges utility that includes bidirectional edges
        start, end = edge_index
        edge_features = self.edge_net(torch.cat([x[start], x[end], topo[start], topo[end]], dim=-1))               
        edge_messages = scatter_add(edge_features, end, dim=0, dim_size=x.shape[0])
        node_inputs = torch.cat([x, topo, edge_messages], dim=-1)
        x_out = self.feature_net(node_inputs)
        x_out += x
        # Now we're ready to repeat the loop...
        
        # 1. MLP(node [features, topo]) -> new topological space
        
        # 2. FRNN in topo space
        
        # 3. Edge convolution with MLP([features, new topo])
        
        # 4. Node aggregation -> new features
        
        # 5. Skip connection with new features and new topo // Optional
        
        # This is the MAXIMAL MIXING configuration, i.e. input -> f(input) = (features, topo) -> f(features, topo) = topo' -> f(features, topo') = edges -> f(edges) = (features', topo') -> f(features, features', topo, topo') = (features, topo)
        
    def message_step(self, x, start, end, e):
        
        # Compute new node features        
        if self.hparams["aggregation"] == "sum":  
            edge_messages = scatter_add(e, end, dim=0, dim_size=x.shape[0]) 
        
        elif self.hparams["aggregation"] == "max":
            edge_messages = scatter_max(e, end, dim=0, dim_size=x.shape[0])[0]

        elif self.hparams["aggregation"] == "sum_max":
            edge_messages = torch.cat([scatter_max(e, end, dim=0, dim_size=x.shape[0])[0],
                                 scatter_add(e, end, dim=0, dim_size=x.shape[0])], dim=-1)
        node_inputs = torch.cat([x, edge_messages], dim=-1)
        
        x_out = self.node_network(node_inputs)
        
        x_out += x

        # Compute new edge features
        edge_inputs = torch.cat([x[start], x[end], e], dim=-1)
        e_out = self.edge_network(edge_inputs)   
        
        e_out += e
        
        return x_out, e_out
        
    def output_step(self, x, batch):
        
        global_pool = global_add_pool(x, batch)
        
        return self.output_graph_regression(global_pool).squeeze(-1)
    
    def forward(self, x, edge_index, batch):

        
        
        start, end = edge_index

        # Encode the graph features into the hidden space
        x = self.node_encoder(x)
        e = self.edge_encoder(torch.cat([x[start], x[end]], dim=1))

        #         edge_outputs = []
        # Loop over iterations of edge and node networks
        for i in range(self.hparams["n_graph_iters"]):

            x, e = self.message_step(x, start, end, e)
        
        # Compute final edge scores; use original edge directions only        
        return self.output_step(x, batch)
        
        
