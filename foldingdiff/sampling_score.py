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
from write_preds_pdb import structure_build_score as structure_build, \
    geometry

# return all frame instead of sidechain rigid frame 
def rigid_apply_update(seq, bb_to_gb, delta_chi, current_local):
    return structure_build.torsion_to_frame(seq, bb_to_gb, delta_chi, current_local)

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
    # [*, N_rigid, 4, 2]
    angles_sin_cos = torch.stack([torch.sin(corrupted_angles), torch.cos(corrupted_angles)], dim=-1)
    default_r = structure_build.get_default_r(seq, corrupted_angles)
    # [*, N_res] Rigid
    bb_to_gb = geometry.get_gb_trans(coords)
    # [*, N_rigid] Rigid
    rigids, current_local_frame,_ = structure_build.torsion_to_frame(seq,
                                                             bb_to_gb,
                                                             angles_sin_cos,
                                                             default_r)

    sigma_schedule = 10 ** torch.linspace(np.log10(sigma_max), np.log10(sigma_min), steps + 1)[:-1]
    eps = 1 / steps

    # Create the attention mask
    pad_mask = torch.zeros(corrupted_angles.shape[:2], device=corrupted_angles.device)
    for i, l in enumerate(seq_lens):
        pad_mask[i, :l] = 1.0
    # print('======angles_sin_cos=====',angles_sin_cos.shape)
    imgs = []
    #for sigma_idx, sigma in enumerate(sigma_schedule):
    for sigma in tqdm(sigma_schedule):

        sigma = torch.unsqueeze(sigma, 0).to('cuda')
        score = model(seq,
                      rigids,
                      sigma,
                      acid_embedding,
                      rigid_type,
                      rigid_property,
                      pad_mask)

        g = sigma * torch.sqrt(torch.tensor(2 * np.log(sigma_max / sigma_min)))
        z = torch.normal(mean=0, std=1, size= score.shape)
        perturb = g.to('cuda') ** 2 * eps * score + g.to('cuda') * np.sqrt(eps) * z.to('cuda')
        perturb_sin_cos = torch.stack((torch.sin(perturb), torch.cos(perturb)), -1)
       # perturb_sin_cos = perturb_sin_cos.view(perturb_sin_cos.shape[0],perturb_sin_cos.shape[-2],perturb_sin_cos.shape[-1]/2,2)
        rigids, current_local_frame, all_frames_to_global = rigid_apply_update(seq, bb_to_gb, perturb_sin_cos, current_local_frame)
        
        torsion_angles = structure_build.rigids_to_torsion_angles(seq, all_frames_to_global)[..., 3:]
        imgs.append(torsion_angles.cpu())

    return torch.stack(imgs)

def sample(
    model: nn.Module,
    train_dset: dsets.NoisedAnglesDataset,
    structure, # a single data dictionary in the dataset
    n: int = 10,
    sweep_lengths: Optional[Tuple[int, int]] = (50, 128),
    batch_size: int = 1,
    feature_key: str = "angles",
    disable_pbar: bool = False,
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
    # Process each batch
    if sweep_lengths is not None:
        sweep_min, sweep_max = sweep_lengths
        logging.info(
            f"Sweeping from {sweep_min}-{sweep_max} with {n} examples at each length"
        )
        lengths = []
        for l in range(sweep_min, sweep_max):
            lengths.extend([l] * n)
    else:
        lengths = [train_dset.sample_length() for _ in range(n)]

    lengths_chunkified = [
        lengths[i : i + batch_size] for i in range(0, len(lengths), batch_size)
    ]

    logging.info(f"Sampling {len(lengths)} items in batches of size {batch_size}")
    retval = []
    temp_c = structure["coords"]
    temp_c = torch.from_numpy(temp_c)
    temp_e = structure["acid_embedding"]
    temp_e = torch.from_numpy(temp_e)
    temp_s = structure["seq"]
    temp_s = torch.from_numpy(temp_s)
    temp_chi = structure["chi_mask"]
    temp_chi = torch.from_numpy(temp_chi)
    temp_rt = structure["rigid_type_onehot"]
    temp_rt = torch.from_numpy(temp_rt)
    temp_rp = structure["rigid_property"]
    temp_rp= torch.from_numpy(temp_rp)
    
    
    for this_lengths in lengths_chunkified:
        torch.cuda.empty_cache()
        batch = len(this_lengths)
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
                          model.n_inputs))

        # Produces (timesteps, batch_size, seq_len, n_ft)
        this_lengths =  [seq.shape[-1], seq.shape[-1], seq.shape[-1], seq.shape[-1],seq.shape[-1], seq.shape[-1], seq.shape[-1], seq.shape[-1],seq.shape[-1], seq.shape[-1]]
        sampled = p_sample_loop_score(model,
                                      coords,
                                      seq,
                                      acid_embedding,
                                      rigid_type,
                                      rigid_property,
                                      this_lengths,
                                      corrupted_angles,
        )

        # Gets to size (timesteps, seq_len, n_ft)
        trimmed_sampled = [
            sampled[:, i, :l, :].numpy() for i, l in enumerate(this_lengths)
        ]

        retval.extend(trimmed_sampled)

    return retval


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