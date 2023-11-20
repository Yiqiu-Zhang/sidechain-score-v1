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

    D_mu = torch.linspace(D_min, D_max, D_count).to(D.device)
    D_mu = D_mu.view([1] * len(D.shape) + [-1])

    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)

    return RBF

def knn_graph(x, k):

    displacement = x[None, :, :] - x[:, None, :]
    distance = torch.linalg.vector_norm(displacement, dim=-1).float()

    # Value of distance [N_rigid, K], Index of distance [N_rigid, K]
    distance, E_idx = torch.topk(distance, k, dim=-1, largest=False)
    col = E_idx.flatten() # source
    row = torch.arange(E_idx.size(0)).view(-1,1).repeat(1,k).flatten().to(col.device) # target

    return torch.stack([row, col], dim=0), distance.flatten()
    
def transform_structure(protein, noise):

    rigids, local_r, all_frames_to_global = structure_build.torsion_to_frame(noise, protein)
    # this is [N_rigid]
    k = 32 if protein.num_nodes >= 32 else protein.num_nodes
    edge_index, distance = knn_graph(rigids.loc, k)

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



    return edge_feature, edge_index, rigids, local_r, all_frames_to_global

class SampleNoiseTransform(BaseTransform):
    def __init__(self, sigma_min=0.01 * np.pi, sigma_max=np.pi):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, data):

        ramdom_sample = torch.distributions.uniform.Uniform(-torch.pi, torch.pi)

        corrupted_angles = ramdom_sample.sample(data.true_chi.shape)
        
        edge_feature, edge_index, rigids, local_r, _ = transform_structure(data, corrupted_angles)

        data.edge_attr = edge_feature
        data.edge_index = edge_index
        data.rigid = rigids
        data.local_rigid = local_r

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(sigma_min={self.sigma_min}, '
                f'sigma_max={self.sigma_max})')

class TorsionNoiseTransform(BaseTransform):
    def __init__(self, sigma_min=0.01 * np.pi, sigma_max=np.pi):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, data):

        sigma = np.exp(np.random.uniform(low=np.log(self.sigma_min), high=np.log(self.sigma_max)))
        sigma = torch.tensor(sigma)
        noise = torch.normal(0, sigma, size=data.true_chi.shape)
        corrupted_angle = data.true_chi + noise

        data.node_sigma = sigma * torch.ones(data.num_nodes)
        data.res_sigma = sigma * torch.ones(data.true_chi.shape)
        data.noise = noise

        edge_feature, edge_index, rigids, local_r, _ = transform_structure(data, corrupted_angle)

        data.edge_attr = edge_feature
        data.edge_index = edge_index
        data.rigid = rigids
        data.local_rigid = local_r

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
