"""
Script to sample from a trained diffusion model
"""
import multiprocessing
import os, sys, shutil
import pickle
import argparse
import logging
from pathlib import Path
from typing import *

import torch
from torch_geometric.loader import DataLoader

sys.path.append(r"/mnt/petrelfs/zhangyiqiu/sidechain-score-v1")

from foldingdiff import modelling_score as modelling
from foldingdiff import sampling_score as sampling
from model import dataset

import glob
sys.path.append(r"../write_preds_pdb")
from structure_build_score import write_pdb_from_position


sys.path.append(r"../foldingdiff")
from ESM1b_embedding import add_esm1b_embedding
from data_pipeline import process_pdb

# :)
SEED = int(
    float.fromhex("54616977616e20697320616e20696e646570656e64656e7420636f756e747279")
    % 10000
)


#srun -p bio_s1 -n 1 --ntasks-per-node=1 --cpus-per-task=20 --gres=gpu:1  python sample.py -m resutl_IPA_Model_6_15 -o IPA_sample_Angle_6.19_casp   


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
    parser.add_argument(
        "-c",
        "--CATH_DIR",
        type=str,
        default="../data/standard_results/native/casp_fixed/",
        help="backbone pdb",
    )

    return parser

def get_pdb_data(CATH_DIR):
    
    # fnames = glob.glob(os.path.join(CATH_DIR, "dompdb", "*"))
    fnames = glob.glob(os.path.join(CATH_DIR, "*"))
    structures = []
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    structures = list(pool.map(process_pdb,fnames, chunksize=250))
    pool.close()
    pool.join()

    return structures

def main() -> None:

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
    
    assert not os.listdir(outdir), f"Expected {outdir} to be empty!"

    plotdir = outdir / "plots"
    os.makedirs(plotdir, exist_ok=True)

    # Load the model
    model_snapshot_dir = outdir / "model_snapshot"
    model = modelling.AngleDiffusionBase.from_dir(
        args.model, copy_to=model_snapshot_dir
    ).to(torch.device(args.device))
    
    torch.manual_seed(args.seed)

    structures = get_pdb_data(args.CATH_DIR)
    structures = add_esm1b_embedding(structures,16)

    test_data_name = '/mnt/petrelfs/zhangyiqiu/sidechain-score-v1/foldingdiff/test_data.pkl'
    with open(test_data_name, "wb") as f:
        pickle.dump(structures, f)

    test_graph_name = '/mnt/petrelfs/zhangyiqiu/sidechain-score-v1/foldingdiff/test_graph.pkl'
    data = dataset.ProteinDataset(cache = test_graph_name, pickle_dir = test_data_name)
    test_loader = DataLoader(dataset=data, batch_size=1)
    
    outdir_pdb = outdir / "sampled_pdb"
    os.makedirs(outdir_pdb, exist_ok=True)

    ramdom_sample = torch.distributions.uniform.Uniform(-torch.pi, torch.pi)

    for protein in test_loader:
        if len(protein.aatype)>128:
            continue

        # [T, N, 4]
        all_atom_positions = sampling.sample(model, protein, ramdom_sample, batch=10)

        # Write the raw sampled items to csv files
        pdbname = Path(protein.fname[0]).name

        for j, atom_positions in enumerate(all_atom_positions):

            write_pdb_from_position(protein, atom_positions, outdir_pdb, pdbname, j)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()