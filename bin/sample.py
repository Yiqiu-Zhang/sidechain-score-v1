"""
Script to sample from a trained diffusion model
"""
import multiprocessing
import os, sys, shutil

import argparse
import logging
import json
from pathlib import Path
from typing import *

import numpy as np
import pandas as pd
import mpl_scatter_density
from matplotlib import pyplot as plt
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize

import torch
from huggingface_hub import snapshot_download

sys.path.append(r"/mnt/petrelfs/zhangyiqiu/sidechain-score-v1")
# Import data loading code from main training script
from train import get_train_valid_test_sets
from annot_secondary_structures import make_ss_cooccurrence_plot

from foldingdiff import modelling_score as modelling
from foldingdiff import sampling_score as sampling
from foldingdiff import plotting
from foldingdiff.datasets_score import AnglesEmptyDataset, NoisedAnglesDataset
from foldingdiff.angles_and_coords import create_new_chain_nerf
from foldingdiff import utils

import glob
sys.path.append(r"../write_preds_pdb")
from structure_build_score import write_preds_pdb_file, write_pdb_from_position


sys.path.append(r"../foldingdiff")
from ESM1b_embedding import add_esm1b_embedding
from data_pipeline import process_pdb

# :)
SEED = int(
    float.fromhex("54616977616e20697320616e20696e646570656e64656e7420636f756e747279")
    % 10000
)

FT_NAME_MAP = {
    "phi": r"$\phi$",
    "psi": r"$\psi$",
    "omega": r"$\omega$",
    "tau": r"$\theta_1$",
    "CA:C:1N": r"$\theta_2$",
    "C:1N:1CA": r"$\theta_3$",
}

#srun -p bio_s1 -n 1 --ntasks-per-node=1 --cpus-per-task=20 --gres=gpu:1  python sample.py -m resutl_IPA_Model_6_15 -o IPA_sample_Angle_6.19_casp   



def build_datasets(
    model_dir: Path, load_actual: bool = True
) -> Tuple[NoisedAnglesDataset, NoisedAnglesDataset, NoisedAnglesDataset]:
    """
    Build datasets given args again. If load_actual is given, the load the actual datasets
    containing actual values; otherwise, load a empty shell that provides the same API for
    faster generation.
    """
    with open(model_dir / "training_args.json") as source:
        training_args = json.load(source)
    # Build args based on training args
    print(training_args)
  #  print("====================training_args[timesteps]========",training_args["timesteps"])
    if True:
        dset_args = dict(
            timesteps=training_args["timesteps"],
            variance_schedule=training_args["variance_schedule"],
            max_seq_len=training_args["max_seq_len"],
            min_seq_len=training_args["min_seq_len"],
            var_scale=training_args["variance_scale"],
            syn_noiser=training_args["syn_noiser"],
            exhaustive_t=training_args["exhaustive_validation_t"],
            single_angle_debug=training_args["single_angle_debug"],
            single_time_debug=training_args["single_timestep_debug"],
            toy=training_args["subset"],
            angles_definitions=training_args["angles_definitions"],
            train_only=False,
        )

        train_dset, valid_dset, test_dset = get_train_valid_test_sets(**dset_args)
        logging.info(
            f"Training dset contains features: {train_dset.feature_names} - angular {train_dset.feature_is_angular}"
        )
        return train_dset, valid_dset, test_dset
    else:
        mean_file = model_dir / "training_mean_offset.npy"
        placeholder_dset = AnglesEmptyDataset(
            feature_set_key=training_args["angles_definitions"],
            pad=training_args["max_seq_len"],
            mean_offset=None if not mean_file.exists() else np.load(mean_file,allow_pickle=True),
        )
        noised_dsets = [
            NoisedAnglesDataset(
                dset=placeholder_dset,
                dset_key="coords"
                if training_args["angles_definitions"] == "cart-coords"
                else "angles",
                timesteps=training_args["timesteps"],
                exhaustive_t=False,
                beta_schedule=training_args["variance_schedule"],
                nonangular_variance=1.0,
                angular_variance=training_args["variance_scale"],
            )
            for _ in range(3)
        ]
        return noised_dsets


def write_preds_pdb_folder(
    final_sampled: Sequence[pd.DataFrame],
    outdir: str,
    basename_prefix: str = "generated_",
    threads: int = multiprocessing.cpu_count(),
) -> List[str]:
    """
    Write the predictions as pdb files in the given folder along with information regarding the
    tm_score for each prediction. Returns the list of files written.
    """
    os.makedirs(outdir, exist_ok=True)
    logging.info(
        f"Writing sampled angles as PDB files to {outdir} using {threads} threads"
    )
    # Create the pairs of arguments
    arg_tuples = [
        (os.path.join(outdir, f"{basename_prefix}{i}.pdb"), samp)
        for i, samp in enumerate(final_sampled)
    ]
    # Write in parallel
    with multiprocessing.Pool(threads) as pool:
        files_written = pool.starmap(create_new_chain_nerf, arg_tuples)

    return files_written


def plot_ramachandran(
    phi_values,
    psi_values,
    fname: str,
    annot_ss: bool = False,
    title: str = "",
    plot_type: Literal["kde", "density_heatmap"] = "density_heatmap",
):
    """Create Ramachandran plot for phi_psi"""
    if plot_type == "kde":
        fig = plotting.plot_joint_kde(
            phi_values,
            psi_values,
        )
        ax = fig.axes[0]
        ax.set_xlim(-3.67, 3.67)
        ax.set_ylim(-3.67, 3.67)
    elif plot_type == "density_heatmap":
        fig = plt.figure(dpi=800)
        ax = fig.add_subplot(1, 1, 1, projection="scatter_density")
        norm = ImageNormalize(vmin=0.0, vmax=650, stretch=LogStretch())
        ax.scatter_density(phi_values, psi_values, norm=norm, cmap=plt.cm.Blues)
    else:
        raise NotImplementedError(f"Cannot plot type: {plot_type}")
    if annot_ss:
        # https://matplotlib.org/stable/tutorials/text/annotations.html
        ram_annot_arrows = dict(
            facecolor="black", shrink=0.05, headwidth=6.0, width=1.5
        )
        ax.annotate(
            r"$\alpha$ helix, LH",
            xy=(1.2, 0.5),
            xycoords="data",
            xytext=(1.7, 1.2),
            textcoords="data",
            arrowprops=ram_annot_arrows,
            horizontalalignment="left",
            verticalalignment="center",
            fontsize=14,
        )
        ax.annotate(
            r"$\alpha$ helix, RH",
            xy=(-1.1, -0.6),
            xycoords="data",
            xytext=(-1.7, -1.9),
            textcoords="data",
            arrowprops=ram_annot_arrows,
            horizontalalignment="right",
            verticalalignment="center",
            fontsize=14,
        )
        ax.annotate(
            r"$\beta$ sheet",
            xy=(-1.67, 2.25),
            xycoords="data",
            xytext=(-0.9, 2.9),
            textcoords="data",
            arrowprops=ram_annot_arrows,
            horizontalalignment="left",
            verticalalignment="center",
            fontsize=14,
        )
    ax.set_xlabel("$\phi$ (radians)", fontsize=14)
    ax.set_ylabel("$\psi$ (radians)", fontsize=14)
    if title:
        ax.set_title(title, fontsize=16)
    fig.savefig(fname, bbox_inches="tight")


def plot_distribution_overlap(
    values_dicts: Dict[str, np.ndarray],
    title: str = "Sampled distribution",
    fname: str = "",
    bins: int = 50,
    ax=None,
    show_legend: bool = True,
    **kwargs,
):
    """
    Plot the distribution overlap between the training and sampled values
    Additional arguments are given to ax.hist; for example, can specify
    histtype='step', cumulative=True
    to get a CDF plot
    """
    # Plot the distribution overlap
    if ax is None:
        fig, ax = plt.subplots(dpi=300)

    for k, v in values_dicts.items():
        if v is None:
            continue
        _n, bins, _pbatches = ax.hist(
            v,
            bins=bins,
            label=k,
            density=True,
            **kwargs,
        )
    if title:
        ax.set_title(title, fontsize=16)
    if show_legend:
        ax.legend()
    if fname:
        fig.savefig(fname, bbox_inches="tight")


def build_parser() -> argparse.ArgumentParser:
    """
    Build CLI parser
    """
    parser = argparse.ArgumentParser(
        usage=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="wukevin/foldingdiff_cath",
        help="Path to model directory, or a repo identifier on huggingface hub. Should contain training_args.json, config.json, and models folder at a minimum.",
    )
    parser.add_argument(
        "--outdir", "-o", type=str, default=os.getcwd(), help="Path to output directory"
    )
    parser.add_argument(
        "--num",
        "-n",
        type=int,
        default=10,
        help="Number of examples to generate *per length*",
    )
    parser.add_argument(
        "-l",
        "--lengths",
        type=int,
        nargs=2,
        default=[127, 128],
        help="Range of lengths to sample from",
    )
    parser.add_argument(
        "-b",
        "--batchsize",
        type=int,
        default=10,
        help="Batch size to use when sampling. 256 consumes ~2GB of GPU memory, 512 ~3.5GB",
    )
    parser.add_argument(
        "--fullhistory",
        action="store_true",
        help="Store full history, not just final structure",
    )
    parser.add_argument(
        "--testcomparison", action="store_true", help="Run comparison against test set"
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    
    #=============================lvying ======================================================
    parser.add_argument(
        "-c",
        "--CATH_DIR",
        type=str,
        default="../data/standard_results/native/casp_fixed/",
        help="backbone pdb",
    )
    #=============================lvying ======================================================

    return parser

#============================================import data========================================

def get_pdb_data(CATH_DIR):
    
    # fnames = glob.glob(os.path.join(CATH_DIR, "dompdb", "*"))
    fnames = glob.glob(os.path.join(CATH_DIR, "*"))
    structures = []
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    structures = list(pool.map(process_pdb,fnames, chunksize=250))
    pool.close()
    pool.join()

    return structures
#============================================import data========================================
def main() -> None:
    """Run the script"""
    parser = build_parser()
    args = parser.parse_args()
    outdir = Path(args.outdir)
    if os.listdir(outdir):
        try:
            print(f"Overwrite {outdir} !!!")
            shutil.rmtree(outdir)
        except:
            print(f"Expected {outdir} to be empty!")
    logging.info(f"Creating {args.outdir}")
    os.makedirs(args.outdir, exist_ok=True)
    
    # Be extra cautious so we don't overwrite any results
    assert not os.listdir(outdir), f"Expected {outdir} to be empty!"

    # Download the model if it was given on modelhub
    #if utils.is_huggingface_hub_id(args.model): #false
    #    logging.info(f"Detected huggingface repo ID {args.model}")
    #    dl_path = snapshot_download(args.model)  # Caching is automatic
    #    assert os.path.isdir(dl_path)
    #    logging.info(f"Using downloaded model at {dl_path}")
    #    args.model = dl_path

    plotdir = outdir / "plots"
    os.makedirs(plotdir, exist_ok=True)

    # Load the model
    model_snapshot_dir = outdir / "model_snapshot"
    print('==========================lvying===================',args.model)
    model = modelling.AngleDiffusionBase.from_dir(
        args.model, copy_to=model_snapshot_dir
    ).to(torch.device(args.device))
    
    torch.manual_seed(args.seed)
    
    #============================================sampling========================================

    structures = get_pdb_data(args.CATH_DIR)
    structures = add_esm1b_embedding(structures,16)
    sampled_angles_folder = outdir / "sampled_angles"
    os.makedirs(sampled_angles_folder, exist_ok=True)
    outdir_pdb = outdir / "sampled_pdb"
    os.makedirs(outdir_pdb, exist_ok=True)
    for structure in structures:
        if len(structure['seq'])>128:
            continue
      #  print("=============fname=================",structure['fname'])
      #  print("=============seq=================",structure['seq'])
        # [B, T, N, 4]
        sampled, all_atom_positions = sampling.sample(model,
                                                    structure,
                                                    batch=args.batchsize,
                                                    )
        # [B, N, 4]
        final_sampled = [s[-1] for s in sampled]
       
        # Write the raw sampled items to csv files
        pdbname = Path(structure['fname']).name

        j = 0
        for atom_positions in all_atom_positions:
            write_pdb_from_position(structure, atom_positions, outdir_pdb, pdbname, j)
            j = j + 1

        '''    
        for sampled_angle in final_sampled: 
            write_preds_pdb_file(structure,sampled_angle, outdir_pdb, pdbname, j)
            j = j+1
        '''
    #============================================sampling========================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()