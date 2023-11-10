"""
Code for sampling from diffusion models
"""
import json
import os
import logging
from typing import *

from tqdm.auto import tqdm

import numpy as np
import pandas as pd

import torch
from torch import nn
from huggingface_hub import snapshot_download

from foldingdiff import beta_schedules
from foldingdiff import utils
from foldingdiff import datasets_score as dsets
from foldingdiff import modelling_score as modelling
from write_preds_pdb import structure_build_score as structure_build


@torch.no_grad()
def p_sample_loop_score(
        model,
        coords:torch.Tensor,
        seq: torch.Tensor,
        acid_embedding: torch.Tensor,
        rigid_type: torch.Tensor,
        rigid_property: torch.Tensor,
        seq_lens: Sequence[int],
        corrupted_angles: torch.Tensor,

        sigma_max = torch.pi,
        sigma_min=0.01 * np.pi,
        steps=100,
):
    # [*, N_rigid] Rigid
    corrupted_sin_cos = torch.stack(
        [torch.sin(corrupted_angles), torch.cos(corrupted_angles)], 
        dim=-1)

    rigids, current_local_r,_ = structure_build.torsion_to_frame(corrupted_sin_cos,
                                                                 seq,
                                                                 coords)

    sigma_schedule = 10 ** torch.linspace(np.log10(sigma_max), np.log10(sigma_min), steps + 1)[:-1].to('cuda')
    eps = 1 / steps

    # Create the attention mask
    pad_mask = torch.zeros(corrupted_angles.shape[:2], device=corrupted_angles.device)
    for i, l in enumerate(seq_lens):
        pad_mask[i, :l] = 1.0

    imgs = []

    for sigma in tqdm(sigma_schedule):

        sigma = torch.unsqueeze(sigma, 0)
        # , current_r
        score = model(rigids, #, _
                    seq,
                    sigma,
                    acid_embedding,
                    rigid_type,
                    rigid_property,
                    pad_mask)
        '''
        if sigma <0.0:
            current_local_r = current_r
            print('updating torsion distance')
        '''
        g = sigma * torch.sqrt(torch.tensor(2 * np.log(sigma_max / sigma_min)))
        z = torch.normal(mean=0, std=1, size= score.shape)
        perturb = g.to('cuda') ** 2 * eps * score + g.to('cuda') * np.sqrt(eps) * z.to('cuda')

        perturb_sin_cos = torch.stack([torch.sin(perturb), torch.cos(perturb)], dim=-1)
        rigids, current_local_r, all_frames_to_global = structure_build.torsion_to_frame(perturb_sin_cos,
                                                                                         seq,
                                                                                         coords,
                                                                                         current_local_r)
            

        corrupted_angles, all_atom_positions = structure_build.rigids_to_torsion_angles(seq, all_frames_to_global)
        imgs.append(corrupted_angles.cpu())
    
    # this is just for convinient use of the backbone coordinate
    # [B, N, 37, 3] [B,N,4,3]
    all_atom_positions[...,:3,:] = coords[...,:3,:]
    all_atom_positions[...,4,:] = coords[...,3,:]

    # Only the final atom position is returned [T,B, N, 4],[B, N, 37, 3]
    return torch.stack(imgs), all_atom_positions

def sample(
    model: nn.Module,
    structure, # a single data dictionary in the dataset
    batch: int = 10,
) -> List[np.ndarray]:
    """
    Sample from the given model. Use the train_dset to generate noise to sample
    sequence lengths. Returns a list of arrays, shape (timesteps, seq_len, fts).
    If sweep_lengths is set, we generate n items per length in the sweep range

    train_dset object must support:
    - sample_noise - provided by NoisedAnglesDataset
    - timesteps - provided by NoisedAnglesDataset
    - alpha_beta_terms - provided by NoisedAnglesDataset
    - feature_is_angular - provided by *wrapped dataset* under NoisedAnglesDataset
    - pad - provided by *wrapped dataset* under NoisedAnglesDataset
    And optionally, sample_length()
    """

    retval = []
    temp_c = structure["coords"]
    temp_c = torch.from_numpy(temp_c)
    temp_e = structure["acid_embedding"]
    temp_e = torch.from_numpy(temp_e)
    temp_s = structure["seq"]
    temp_s = torch.from_numpy(temp_s)
    temp_chi = structure["chi_mask"]
    temp_rt = structure["rigid_type_onehot"]
    temp_rt = torch.from_numpy(temp_rt)
    temp_rp = structure["rigid_property"]
    temp_rp= torch.from_numpy(temp_rp)
    
    
    torch.cuda.empty_cache()
    # batch is always one, should we set this batch size to N
    # so we can generate N sample at a single time ?????????
    # Sample noise and sample the lengths
    coords = temp_c.repeat(batch,1,1,1).cuda()
    acid_embedding = temp_e.repeat(batch,1,1).cuda()
    seq = temp_s.unsqueeze(1).repeat(batch,1,1).cuda().squeeze(-1)
    chi_mask = temp_chi.repeat(batch,1,1).cuda()
    rigid_type = temp_rt.repeat(batch,1,1,1).cuda()
    rigid_property = temp_rp.repeat(batch,1,1,1).cuda()

    # [b,n,f]
    m = torch.distributions.uniform.Uniform(-torch.pi, torch.pi)
    corrupted_angles = m.sample((batch,
                        seq.shape[-1], # sequence length
                        ))

    # Produces (timesteps, batch_size, seq_len, n_ft)
    this_lengths =  [seq.shape[-1], seq.shape[-1], seq.shape[-1], seq.shape[-1],seq.shape[-1], seq.shape[-1], seq.shape[-1], seq.shape[-1],seq.shape[-1], seq.shape[-1]]
    sampled, all_atom_positions = p_sample_loop_score(model,
                                    coords,
                                    seq,
                                    acid_embedding,
                                    rigid_type,
                                    rigid_property,
                                    this_lengths,
                                    corrupted_angles,
    )
    # [B, T, N, 4]
    # Gets to size (timesteps, seq_len, n_ft)
    trimmed_sampled = [
        sampled[:, i, :l, :].numpy() for i, l in enumerate(this_lengths)
    ]

    retval.extend(trimmed_sampled)

    return retval, all_atom_positions


def sample_simple(
    model_dir: str, n: int = 10, sweep_lengths: Tuple[int, int] = (50, 128)
) -> List[pd.DataFrame]:
    """
    Simple wrapper on sample to automatically load in the model and dummy dataset
    Primarily for gradio integration
    """
    if utils.is_huggingface_hub_id(model_dir):
        model_dir = snapshot_download(model_dir)
    assert os.path.isdir(model_dir)

    with open(os.path.join(model_dir, "training_args.json")) as source:
        training_args = json.load(source)

    model = modelling.BertForDiffusionBase.from_dir(model_dir)
    if torch.cuda.is_available():
        model = model.to("cuda:0")

    dummy_dset = dsets.AnglesEmptyDataset.from_dir(model_dir)
    dummy_noised_dset = dsets.NoisedAnglesDataset(
        dset=dummy_dset,
        dset_key="coords" if training_args == "cart-cords" else "angles",
        timesteps=training_args["timesteps"],
        exhaustive_t=False,
        beta_schedule=training_args["variance_schedule"],
        nonangular_variance=1.0,
        angular_variance=training_args["variance_scale"],
    )

    sampled = sample(model, dummy_noised_dset, n=n, sweep_lengths=sweep_lengths, disable_pbar=True)
    final_sampled = [s[-1] for s in sampled]
    sampled_dfs = [
        pd.DataFrame(s, columns=dummy_noised_dset.feature_names["angles"])
        for s in final_sampled
    ]
    return sampled_dfs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    s = sample_simple("wukevin/foldingdiff_cath", n=1, sweep_lengths=(50, 55))
    for i, x in enumerate(s):
        print(x.shape)
