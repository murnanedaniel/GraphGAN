import os, sys
import logging

import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
import scipy as sp
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
# import cupy as cp
# import trackml.dataset



# ---------------------------- Dataset Processing -------------------------



def load_dataset(datasplit):
    
    all_datasets = []
    
    for data_num in datasplit:
                
        # tracks = process_map(run_generation, range(data_num), num_workers=4)

        tracks = []

        for i in tqdm(range(data_num)):
            tracks.append(run_generation(i))
            
        all_datasets.append(tracks)
        
    return all_datasets

def run_generation(i):
    generated = False
    while not generated:
        nodes, edges, y = generate_merged_track()
        if (nodes == nodes).all():
            generated = True
    return Data(x=nodes, edge_index=edges, y=y)

def calc_y(x, r, theta, sign):
    return sign*(np.sqrt(r**2 - (x - r*np.cos(theta))**2) - r*np.sin(theta))

def generate_circle(r = None):
    if r is None:
        r = np.random.uniform(1, 5)
    theta = np.random.uniform(0, np.pi)
    sign = np.random.choice([-1, 1])
    return r, theta, sign

def generate_merged_track(detector_width = 1, num_layers = 4):
    track_1_params = generate_circle()
    track_2_params = generate_circle()

    x = np.linspace(0, detector_width, num = num_layers)
    track_1_y = calc_y(x, *track_1_params)
    track_2_y = calc_y(x[1:], *track_2_params)

    track_1_edges = np.vstack([np.arange(len(track_1_y)-1), np.arange(1, len(track_1_y))])
    track_2_edges = np.vstack([np.arange(len(track_2_y)-1), np.arange(1, len(track_2_y))]) + len(track_1_y)

    track_1_nodes = np.vstack([x, track_1_y]).T
    track_2_nodes = np.vstack([x[1:], track_2_y]).T

    nodes = torch.from_numpy(np.concatenate([track_1_nodes, track_2_nodes]))
    true_edges = torch.from_numpy(np.concatenate([track_1_edges, track_2_edges], axis = -1))
       
    FC_edges = torch.combinations(torch.arange(0, len(nodes)), r=2).T
    FC_edges, y = graph_intersection(FC_edges, torch.cat([true_edges, true_edges.flip(0)], axis=1))
    
    return nodes, FC_edges, y
    
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
            layers.append(nn.BatchNorm1d(sizes[i + 1], track_running_stats=False))
        layers.append(hidden_activation())
    # Final layer
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    if output_activation is not None:
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[-1]))
        if batch_norm:
            layers.append(nn.BatchNorm1d(sizes[i + 1], track_running_stats=False))
        layers.append(output_activation())
    return nn.Sequential(*layers)
