import multiprocessing
import os, sys, shutil
import pickle
import argparse
import logging
from pathlib import Path
from typing import *
import copy

import torch
from torch_geometric.loader import DataLoader

sys.path.append(r"/mnt/petrelfs/zhangyiqiu/sidechain-score-v1")

from foldingdiff import modelling_score as modelling
from foldingdiff import sampling_score as sampling
from model import dataset

import glob
from write_preds_pdb.structure_build_score import write_pdb_from_position, torsion_to_frame, frame_to_pos


from foldingdiff.ESM1b_embedding import add_esm1b_embedding
from foldingdiff.data_pipeline import process_pdb

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

    #test_data_name = '/mnt/petrelfs/zhangyiqiu/sidechain-score-v1/foldingdiff/test_data.pkl'
    test_graph_name = '/mnt/petrelfs/zhangyiqiu/sidechain-score-v1/foldingdiff/test_graph.pkl'
    graph_data = dataset.preprocess_datapoints(graph_data = test_graph_name)

    sample_transform = dataset.SampleNoiseTransform()
    data = dataset.ProteinDataset(data = graph_data, transform=sample_transform) #, pickle_dir = test_data_name
    
    outdir_pdb = outdir / "sampled_pdb"
    os.makedirs(outdir_pdb, exist_ok=True)

    for i in range(5):
        for protein in data[:1]:
            
            if len(protein.aatype)>128:
                continue
            
            prot_gpu = copy.deepcopy(protein).to('cuda')
            pdbname = Path(protein.fname).name

            all_atom_positions = sampling.p_sample_loop_score(model, prot_gpu)            
            write_pdb_from_position(protein, all_atom_positions, outdir_pdb, pdbname, i)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()