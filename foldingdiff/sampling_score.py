"""
Code for sampling from diffusion models
"""
import logging
from typing import *

from tqdm.auto import tqdm

import numpy as np

import torch
from torch import nn

from write_preds_pdb import structure_build_score as structure_build
from model.data_preprocessing import protein_to_graph
from model.dataset import transform_structure


@torch.no_grad()
def p_sample_loop_score(
        model,
        protein,
        sigma_max = torch.pi,
        sigma_min=0.01 * np.pi,
        steps=100,
):

    sigma_schedule = 10 ** torch.linspace(np.log10(sigma_max), np.log10(sigma_min), steps + 1)[:-1].to('cuda')
    eps = 1 / steps

    for sigma in tqdm(sigma_schedule):

        protein.node_sigma = sigma * torch.ones(protein.num_nodes).to('cuda')

        score = model(protein)

        g = sigma * torch.sqrt(torch.tensor(2 * np.log(sigma_max / sigma_min))).to('cuda')
        z = torch.normal(mean=0, std=1, size= score.shape).to('cuda')
        perturb = g ** 2 * eps * score + g * np.sqrt(eps) * z

        perturb_sin_cos = torch.stack([torch.sin(perturb), torch.cos(perturb)], dim=-1).to('cuda')
        rigid, local_rigid, all_frames_to_global = structure_build.torsion_to_frame(perturb_sin_cos, protein)
        protein.rigid = rigid
        protein.local_rigid = local_rigid
            
    all_atom_positions = structure_build.rigids_to_torsion_angles(protein.aatype, all_frames_to_global)

    # this is just for convinient use of the backbone coordinate
    # [B, N, 37, 3] [B,N,4,3]
    all_atom_positions[...,:3,:] = protein.bb_coord[...,:3,:]
    all_atom_positions[...,4,:] = protein.bb_coord[...,3,:]

    return all_atom_positions

def sample(
    model: nn.Module,
    protein, # a single data dictionary in the dataset
    ramdom_sample,
    batch: int = 10,
) -> List[np.ndarray]:

    torch.cuda.empty_cache()

    protein.true_chi = protein.true_chi.cuda()
    protein.aatype = protein.aatype.cuda()
    protein.bb_coord = protein.bb_coord.cuda()
    protein.res_index = protein.res_index.cuda()
    protein.chi_mask = protein.chi_mask.cuda()
    protein.rigid_mask = protein.rigid_mask.cuda()
    protein.x = protein.x.cuda()

    batched_position = []

    for _ in range(batch):

        noise = ramdom_sample.sample(protein.true_chi.shape).cuda()
        protein = transform_structure(protein, noise)

        protein.edge_attr = protein.edge_attr.cuda()
        protein.edge_index = protein.edge_index.cuda()
        protein.rigid = protein.rigid.cuda()
        protein.local_rigid = protein.local_rigid.cuda()
        
        # Produces (timesteps, batch_size, seq_len, n_ft)
        all_atom_positions = p_sample_loop_score(model, protein)

        batched_position.append(all_atom_positions)

    return batched_position

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
