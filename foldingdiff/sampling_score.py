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
        sigma_max = torch.pi,
        sigma_min=0.01 * np.pi,
        steps=100,
):
    
    sigma_schedule = 10 ** torch.linspace(np.log10(sigma_max), np.log10(sigma_min), steps + 1)[:-1].to('cuda')
    eps = 1 / steps

    for sigma in sigma_schedule:

        protein.node_sigma = sigma * torch.ones(protein.num_nodes).to('cuda')

        score = model(protein)

        g = sigma * torch.sqrt(torch.tensor(2 * np.log(sigma_max / sigma_min))).to('cuda')
        z = torch.normal(mean=0, std=1, size= score.shape).to('cuda')
        perturb = g ** 2 * eps * score + g * np.sqrt(eps) * z

        protein, all_frames_to_global = transform_structure(protein, perturb)

    all_atom_positions = structure_build.frame_to_pos(all_frames_to_global, 
                                                      protein.aatype,
                                                      protein.bb_coord)

    return all_atom_positions

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
