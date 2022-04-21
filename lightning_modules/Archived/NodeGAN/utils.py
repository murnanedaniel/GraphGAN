import os, sys
import logging

import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
# import cupy as cp
# import trackml.dataset


# ---------------------------- Dataset Processing -------------------------



def load_dataset(datasplit):
    
    all_datasets = []
    
    for data_num in datasplit:
        
        polygons = []

        for i in range(data_num):

            nodes, edges, N = generate_polygon()
            polygons.append(Data(x=nodes, edge_index=edges, y=torch.tensor(N)))
        
        all_datasets.append(polygons)
        
    return all_datasets

def rotate(point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    px, py = point

    qx = np.cos(angle) * (px) - np.sin(angle) * (py)
    qy = np.sin(angle) * (px) + np.cos(angle) * (py)
    return np.round(qx, 2), np.round(qy, 2)

def generate_polygon(N = None):
    if N is None:
        N = np.random.randint(3, 7)
    R = np.random.rand()
    x0, y0 = np.random.rand(2)
    theta0 = np.random.rand()*np.pi
    
    seed_point = rotate(np.array([0, R]), theta0)
    O = np.array([x0, y0])
    nodes = np.tile(seed_point, (N, 1)).astype(float)
    rotations = np.linspace(0, 2*np.pi, N+1)[:-1]
    
    for i, rotation in enumerate(rotations):
        nodes[i] = rotate(nodes[i], rotation)
        nodes[i] = nodes[i] + O
    
    edges = np.stack([np.arange(0, N), np.roll(np.arange(0, N), 1)])
    
    return torch.from_numpy(nodes), torch.from_numpy(edges), N
    

# ------------------------- Convenience Utilities ---------------------------


def make_mlp(
    input_size,
    sizes,
    hidden_activation="ReLU",
    output_activation="ReLU",
    layer_norm=False,
):
    """Construct an MLP with specified fully-connected layers."""
    hidden_activation = getattr(nn, hidden_activation)
    if output_activation is not None:
        output_activation = getattr(nn, output_activation)
    layers = []
    n_layers = len(sizes)
    sizes = [input_size] + sizes
    # Hidden layers
    for i in range(n_layers - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[i + 1]))
        layers.append(hidden_activation())
    # Final layer
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    if output_activation is not None:
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[-1]))
        layers.append(output_activation())
    return nn.Sequential(*layers)
