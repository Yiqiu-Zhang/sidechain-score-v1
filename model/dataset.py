import copy

from data_preprocessing import protein_to_graph
from write_preds_pdb.structure_build_score import torsion_to_frame
from write_preds_pdb.geometry import to_tensor_5x5

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

    D_mu = torch.linspace(D_min, D_max, D_count).to(D)
    D_mu = D_mu.view([1] * len(D.shape) + [-1])

    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)

    return RBF

def knn_graph(x, k):

    displacement = x[None, :, :] - x[:, None, :]
    distance = torch.linalg.vector_norm(displacement, dim=-1).float().to(x)

    # Value of distance [N_rigid, K], Index of distance [N_rigid, K]
    distance, E_idx = torch.topk(distance, k, dim=-1, largest=False)
    col = E_idx.flatten() # source
    row = torch.arange(E_idx.size(0)).view(-1,1).repeat(1,k).flatten().to(col) # target

    return torch.stack([row, col], dim=0), distance.flatten()
    
def transform_structure(protein, noise):

    rigids, local_r, all_frames_to_global = torsion_to_frame(noise, protein)
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
    
    protein.edge_attr = edge_feature
    protein.edge_index = edge_index
    protein.rigid = to_tensor_5x5(rigids)
    protein.local_rigid = to_tensor_5x5(local_r)
    del local_r
    del rigids

    return protein, to_tensor_5x5(all_frames_to_global)

class SampleNoiseTransform(BaseTransform):
    def __init__(self):

        self.ramdom_sample = torch.distributions.uniform.Uniform(-torch.pi, torch.pi)

    def __call__(self, protein):

        corrupted_angles = self.ramdom_sample.sample(protein.true_chi.shape)
        protein, _ = transform_structure(protein, corrupted_angles)

        return protein

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}')

class TorsionNoiseTransform(BaseTransform):
    def __init__(self, sigma_min=0.01 * np.pi, sigma_max=np.pi):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, protein):
        sigma = np.exp(np.random.uniform(low=np.log(self.sigma_min), high=np.log(self.sigma_max)))
        sigma = torch.tensor(sigma).to(protein.true_chi.device)
        noise = torch.normal(0, sigma, size=protein.true_chi.shape).to(protein.true_chi.device)
        corrupted_angle = protein.true_chi + noise

        protein.node_sigma = sigma * torch.ones(protein.num_nodes)
        protein.res_sigma = sigma * torch.ones(protein.true_chi.shape)
        protein.noise = noise
        
        protein, _ = transform_structure(protein, corrupted_angle)

        return protein

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(sigma_min={self.sigma_min}, '
                f'sigma_max={self.sigma_max})')

class ProteinDataset(Dataset):
    def __init__(self,data, transform = None):

        super(ProteinDataset, self).__init__(transform)
        self.transform = transform
        self.proteins = data

    def len(self): return len(self.proteins)
    def get(self, item):
        protein =  self.proteins[item]
        return copy.deepcopy(protein)

def preprocess_datapoints(graph_data=None, raw_dir=None):

    if graph_data and os.path.exists(graph_data):
        print('Reusing graph_data', graph_data)
        with open(graph_data, "rb") as f:
            proteins = pickle.load(f)
    else:
        print("Preprocessing")
        with open(raw_dir, "rb") as file:
            proteins_list = pickle.load(file)

        proteins = []
        for protein in proteins_list:
            proteins.append(protein_to_graph(protein))

        
        if graph_data:
            print("Store_data at", graph_data)
            with open(graph_data, "wb") as f:
                pickle.dump(proteins, f)
                print('finish data preprocess')
                
    return proteins