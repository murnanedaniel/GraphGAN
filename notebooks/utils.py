import numpy as np
import torch

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
    
    edges = np.stack([np.arange(0, N), np.roll(np.arange(0, N), 1)])
    
    return torch.from_numpy(nodes), torch.from_numpy(edges), N