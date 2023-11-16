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
from model.dataset import transform_structure


@torch.no_grad()
def p_sample_loop_score(
        model,
        protein,
        ramdom_sample,
        sigma_max = torch.pi,
        sigma_min=0.01 * np.pi,
        steps=100,
):
    
    sigma_schedule = 10 ** torch.linspace(np.log10(sigma_max), np.log10(sigma_min), steps + 1)[:-1].to('cuda')
    eps = 1 / steps

    batched_position = []

    for _ in range(10):

        corrupted_angles = ramdom_sample.sample(protein.true_chi.shape).cuda()
        protein = transform_structure(protein, corrupted_angles)

        protein.edge_attr = protein.edge_attr.cuda()
        protein.edge_index = protein.edge_index.cuda()
        protein.rigid = protein.rigid.cuda()
        protein.local_rigid = protein.local_rigid.cuda()

        for sigma in tqdm(sigma_schedule):

            protein.node_sigma = sigma * torch.ones(protein.num_nodes).to('cuda')

            score = model(protein)

            g = sigma * torch.sqrt(torch.tensor(2 * np.log(sigma_max / sigma_min))).to('cuda')
            z = torch.normal(mean=0, std=1, size= score.shape).to('cuda')
            perturb = g ** 2 * eps * score + g * np.sqrt(eps) * z

            protein = transform_structure(protein, perturb)

        all_atom_positions = structure_build.frame_to_pos(protein.all_frames_to_global, 
                                                          protein.aatype,
                                                          protein.bb_coord)

        batched_position.append(all_atom_positions)

    return batched_position

def sample(
    model: nn.Module,
    protein, # a single data dictionary in the dataset
    ramdom_sample,
) -> List[np.ndarray]:

    torch.cuda.empty_cache()

    protein.true_chi = protein.true_chi.cuda()
    protein.aatype = protein.aatype.cuda()
    protein.bb_coord = protein.bb_coord.cuda()
    protein.res_index = protein.res_index.cuda()
    protein.chi_mask = protein.chi_mask.cuda()
    protein.rigid_mask = protein.rigid_mask.cuda()
    protein.x = protein.x.cuda()

    batched_position = p_sample_loop_score(model, protein, ramdom_sample)

    return batched_position

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
