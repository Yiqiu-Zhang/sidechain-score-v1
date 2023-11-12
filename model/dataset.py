import copy

from data_preprocessing import protein_to_graph
import write_preds_pdb.structure_build_score as structure_build

import torch
import numpy as np
import os
import pickle
from torch import nn
from torch_geometric.data import Dataset, DataLoader, lightning
from torch_geometric.transforms import BaseTransform
import scipy

def relpos(rigid_res_index, edge_index):

    d_i = rigid_res_index[edge_index[0]]
    d_j = rigid_res_index[edge_index[1]]

    # [E]
    d = d_i - d_j

    boundaries = torch.arange(start=-32, end=32 + 1, device=d.device)
    reshaped_bins = boundaries.view(1, len(boundaries))

    d = d[..., None] - reshaped_bins
    d = torch.abs(d)
    d = torch.argmin(d, dim=-1)
    d = nn.functional.one_hot(d, num_classes=len(boundaries)).float()

    return d # [N_rigid, relpos_k]

def rbf(D, D_min=0., D_max=20., D_count=16):
    # Distance radial basis function

    D_mu = torch.linspace(D_min, D_max, D_count)
    D_mu = D_mu.view([1] * len(D.shape) + [-1])

    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)

    return RBF

def knn_graph(x, k):

    batch_x = x.new_zeros(x.size(0), dtype=torch.long)

    # Rescale x and y.
    min_x = x.min().item()
    x  = x - min_x

    max_x = x.max().item()
    x = x / max_x

    x = torch.cat([x, 2 * x.size(1) * batch_x.view(-1, 1).to(x.dtype)], dim=-1)

    tree = scipy.spatial.cKDTree(x.detach().cpu().numpy())

    dist, col = tree.query(
        x.detach().cpu(), k=k, distance_upper_bound=x.size(1))
    
    dist = torch.from_numpy(dist).to(x.dtype)
    col = torch.from_numpy(col).to(torch.long)
    row = torch.arange(col.size(0), dtype=torch.long).view(-1, 1).repeat(1, k)
    mask = 1 - torch.isinf(dist).view(-1).long()
    row, col = row.view(-1)[mask], col.view(-1)[mask]

    return torch.stack([row, col], dim=0)
    
def transform_structure(protein, noise):

    # Edge_feature
    angle_sin_cos = torch.stack([torch.sin(noise), torch.cos(noise)], dim=-1)

    rigids = structure_build.torsion_to_frame(angle_sin_cos, protein)
    # this is [N_rigid]

    edge_index = knn_graph(rigids.loc, k=32)

    distance, altered_direction, orientation = rigids.edge(edge_index)

    distance_rbf = rbf(distance)
    rigid_res_idx = protein.res_index.unsqueeze(-1).repeat(1, 5).reshape(-1)
    rigid_res_idx = rigid_res_idx[protein.rigid_mask]

    relative_pos = relpos(rigid_res_idx, edge_index)
    nf_pair_feature = torch.cat([protein.x[edge_index[0]], protein.x[edge_index[1]]], axis=-1)

    edge_feature = torch.cat([distance_rbf,
                              nf_pair_feature,
                              relative_pos,
                              ],
                             dim=-1).float()

    protein.edge_attr = edge_feature
    protein.edge_index = edge_index
    protein.rigid = rigids

    return protein

class TorsionNoiseTransform(BaseTransform):
    def __init__(self, sigma_min=0.01 * np.pi, sigma_max=np.pi):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, data):

        sigma = np.exp(np.random.uniform(low=np.log(self.sigma_min), high=np.log(self.sigma_max)))
        sigma = torch.tensor(sigma)
        noise = torch.normal(0, sigma, size=data.true_chi.shape)

        data.node_sigma = sigma * torch.ones(data.num_nodes)
        data.res_sigma = sigma * torch.ones(data.true_chi.shape)
        data.noise = noise

        data = transform_structure(data, noise)

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(sigma_min={self.sigma_min}, '
                f'sigma_max={self.sigma_max})')

class ProteinDataset(Dataset):
    def __init__(self,cache=None, pickle_dir=None, split=None, transform = None):

        super(ProteinDataset, self).__init__(transform)
        self.transform = transform
        if cache and os.path.exists(cache):
            print('Reusing preprocessing from cache', cache)
            with open(cache, "rb") as f:
                self.proteins = pickle.load(f)
        else:
            print("Preprocessing")
            self.proteins = self.preprocess_datapoints(pickle_dir)
            if cache:
                print("Caching at", cache)
                with open(cache, "wb") as f:
                    pickle.dump(self.proteins, f)

        if split is not None:
            split_idx = int(len(self.proteins) * 0.9)
            if split == "train":
                self.proteins = self.proteins[:split_idx]
            elif split == "validation":
                self.proteins = self.proteins[split_idx: split_idx + int(len(self.structures) * 0.1)]

    def len(self): return len(self.proteins)
    def get(self, item):
        protein =  self.proteins[item]
        return copy.deepcopy(protein)

    def preprocess_datapoints(self, dir):

        with open(dir, "rb") as file:
            proteins_list = pickle.load(file)

        graph = []
        for protein in proteins_list:
            graph.append(protein_to_graph(protein))

        return graph
