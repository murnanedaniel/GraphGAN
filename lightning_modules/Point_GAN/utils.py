import os, sys
import logging

import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
import scipy as sp
from tqdm import tqdm
# import cupy as cp
# import trackml.dataset


# ---------------------------- Dataset Processing -------------------------



def load_dataset(datasplit):
    
    all_datasets = []
    
    for data_num in datasplit:
        
        polygons = []

        for i in tqdm(range(data_num)):

            nodes, edges, edge_scores, N = generate_polygon(4)
            polygons.append(Data(x=nodes, edge_index=edges, edge_attr=edge_scores.float(), y=torch.tensor(N)))
        
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
        N = np.random.randint(4, 8)
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
    
    edges = torch.from_numpy(np.stack([np.arange(0, N), np.roll(np.arange(0, N), 1)]))
    
    FC_edges = torch.combinations(torch.arange(0, N), r=2).T
    FC_edges, edge_scores = graph_intersection(FC_edges, torch.cat([edges, edges.flip(0)], axis=1))
    
    return torch.from_numpy(nodes), FC_edges, edge_scores, N
    
def graph_intersection(
    pred_graph, truth_graph
):

    array_size = max(pred_graph.max().item(), truth_graph.max().item()) + 1

    if torch.is_tensor(pred_graph):
        l1 = pred_graph.cpu().numpy()
    else:
        l1 = pred_graph
    if torch.is_tensor(truth_graph):
        l2 = truth_graph.cpu().numpy()
    else:
        l2 = truth_graph
    e_1 = sp.sparse.coo_matrix(
        (np.ones(l1.shape[1]), l1), shape=(array_size, array_size)
    ).tocsr()
    e_2 = sp.sparse.coo_matrix(
        (np.ones(l2.shape[1]), l2), shape=(array_size, array_size)
    ).tocsr()
    del l1

    e_intersection = e_1.multiply(e_2) - ((e_1 - e_2) > 0)

    e_intersection = e_intersection.tocoo()
    new_pred_graph = torch.from_numpy(
        np.vstack([e_intersection.row, e_intersection.col])
    ).long()  # .to(device)
    y = torch.from_numpy(e_intersection.data > 0)  # .to(device)

    return new_pred_graph, y
    
# ------------------------- Convenience Utilities ---------------------------


def make_mlp(
    input_size,
    sizes,
    hidden_activation="ReLU",
    output_activation="ReLU",
    layer_norm=False,
    batch_norm=False,
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
        if batch_norm:
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
        layers.append(hidden_activation())
    # Final layer
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    if output_activation is not None:
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[-1]))
        if batch_norm:
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
        layers.append(output_activation())
    return nn.Sequential(*layers)
