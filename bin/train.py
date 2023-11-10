"""
Training script.

Example usage: python ~/protdiff/bin/train.py ~/protdiff/config_jsons/full_run_canonical_angles_only_zero_centered_1000_timesteps_reduced_len.json
srun -p bio_s1 -n 1 --ntasks-per-node=1 --cpus-per-task=40 --gres=gpu:2 python train.py /mnt/petrelfs/lvying/code/sidechain-rigid-attention/config_jsons/cath_full_angles_cosine.json --dryrun
squeue -p bio_s1
sbatch -p bio_s1 --ntasks-per-node=1 --cpus-per-task=10 --gres=gpu:1  sample_8.7.sh
sbatch -p bio_s1 --ntasks-per-node=1 --cpus-per-task=64 --gres=gpu:7  IPA_Score_8.3.sh 
export http_proxy="http://zhangyiqiu:Wzdhxzh5bn2023@10.1.8.50:33128"
export https_proxy="http://zhangyiqiu:Wzdhxzh5bn2023@10.1.8.50:33128"
swatch  -n  SH-IDC1-10-140-1-157  nv
请问如何查看之前完成的任务，用什么命令。 sacct -u  ad账号
"""

import os, sys
import shutil
import json
import logging
from pathlib import Path
import multiprocessing
import argparse
import functools
from datetime import datetime
from typing import *

import numpy as np
from matplotlib import pyplot as plt

import torch
from torch.utils.data import Dataset, Subset
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
from torch_geometric.data import Dataset, DataLoader, lightning

from transformers import BertConfig

sys.path.append(r"/mnt/petrelfs/zhangyiqiu/sidechain-score-v1")
#from foldingdiff import datasets_score as datasets
from foldingdiff import modelling_score as modelling
from foldingdiff import losses_score as losses
from foldingdiff import beta_schedules
from foldingdiff import plotting
from foldingdiff import utils
from foldingdiff import custom_metrics as cm
from model import dataset
from torchsummary import summary

#                        
#from pytorch_lightning.profiler import SimpleProfiler, AbstractProfiler, AdvancedProfiler, PyTorchProfiler
from pytorch_lightning.loggers import TensorBoardLogger

assert torch.cuda.is_available(), "Requires CUDA to train"
# reproducibility
torch.manual_seed(6489)
# torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False

# Define some typing literals
ANGLES_DEFINITIONS = Literal[
    "canonical", "canonical-full-angles", "canonical-minimal-angles", "cart-coords","side-chain-angles"
]


@pl.utilities.rank_zero_only
def plot_timestep_distributions(
    train_dset,
    timesteps: int,
    plots_folder: Path,
    shift_angles_zero_twopi: bool = False,
    n_intervals: int = 11,
) -> None:
    """
    Plot the distributions across timesteps. This is parallelized across multiple cores
    """
    ts = np.linspace(0, timesteps, num=n_intervals, endpoint=True).astype(int)
    ts = np.minimum(ts, timesteps - 1).tolist()
    logging.info(f"Plotting distributions at {ts} to {plots_folder}")
    args = [
        (
            t,
            train_dset,
            True,
            not shift_angles_zero_twopi,
            plots_folder / f"train_dists_at_t_{t}.pdf",
        )
        for t in ts
    ]

    # Parallelize the plotting
    pool = multiprocessing.Pool(processes=min(multiprocessing.cpu_count(), len(ts)))
    pool.starmap(plotting.plot_val_dists_at_t, args)
    pool.close()
    pool.join()


@pl.utilities.rank_zero_only
def plot_kl_divergence(train_dset, plots_folder: Path) -> None:
    """
    Plot the KL divergence over time
    """
    # This works because the main body of this script should clean out the dir
    # between runs
    outname = plots_folder / "kl_divergence_timesteps.pdf"
    if outname.is_file():
        logging.info(f"KL divergence plot exists at {outname}; skipping...")
    kl_at_timesteps = cm.kl_from_dset(train_dset)  # Shape (n_timesteps, n_features)
    n_timesteps, n_features = kl_at_timesteps.shape
    fig, axes = plt.subplots(
        dpi=300, figsize=(n_features * 3.05, 2.5), ncols=n_features, sharey=True
    )
    for i, (ft_name, ax) in enumerate(zip(train_dset.feature_names["angles"], axes)):
        ax.plot(np.arange(n_timesteps), kl_at_timesteps[:, i], label=ft_name)
        ax.axhline(0, color="grey", linestyle="--", alpha=0.5)
        ax.set(title=ft_name)
        if i == 0:
            ax.set(ylabel="KL divergence")
        ax.set(xlabel="Timestep")
    fig.suptitle(
        f"KL(empirical || Gaussian) over timesteps={train_dset.timesteps}", y=1.05
    )
    fig.savefig(outname, bbox_inches="tight")

def build_callbacks(outdir: str):
    """
    Build out the callbacks
    """
    # Create the logging dir
    os.makedirs(os.path.join(outdir, "logs/lightning_logs"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "models/best_by_valid"), exist_ok=True)

    callbacks = [pl.callbacks.ModelCheckpoint(
                    monitor="val_loss",
                    dirpath=os.path.join(outdir, "models/best_by_valid"),
                    filename='sample-mnist-{epoch:02d}-{val_loss:.4f}',
                    save_top_k=1,
                    save_weights_only=False,
                    mode="min",
                    save_on_train_epoch_end = False,
                    save_last=True,
                )]

    logging.info(f"Model callbacks: {callbacks}")
    return callbacks


# For some arg defaults, see as reference:
# https://huggingface.co/docs/transformers/main/en/main_classes/trainer.html


@pl.utilities.rank_zero_only
def record_args_and_metadata(func_args: Dict[str, Any], results_folder: Path):
    # Create results directory
    if results_folder.exists():
        logging.warning(f"Removing old results directory: {results_folder}")
        shutil.rmtree(results_folder)
    results_folder.mkdir(exist_ok=True)
    with open(results_folder / "training_args.json", "w") as sink:
        logging.info(f"Writing training args to {sink.name}")
        json.dump(func_args, sink, indent=4)
        for k, v in func_args.items():
            logging.info(f"Training argument: {k}={v}")

def train(
    # Controls output
    results_dir: str = "./results",
    # Controls data loading and noising process
    dataset_key: str = "bc40",  # cath, alhpafold, or a directory containing pdb files

    # Related to training strategy
    gradient_clip: float = 1.0,  # From BERT trainer
    batch_size: int = 16,
    lr: float = 5e-5,  # Default lr for huggingface BERT trainer
    l2_norm: float = 0.0,  # AdamW default has 0.01 L2 regularization, but BERT trainer uses 0.0
    l1_norm: float = 0.0,
    min_epochs: Optional[int] = None,
    max_epochs: int = 10000,
    lr_scheduler: modelling.LR_SCHEDULE = None,

    cpu_only: bool = False,
    ngpu: int = -1,  # -1 for all GPUs
    write_valid_preds: bool = False,  # Write validation predictions to disk at each epoch

):
    """Main training loop"""
    # Record the args given to the function before we create more vars
    # https://stackoverflow.com/questions/10724495/getting-all-arguments-and-values-passed-to-a-function
    func_args = locals()

    results_folder = Path(results_dir)
    record_args_and_metadata(func_args, results_folder)

    graph_data = '/mnt/petrelfs/zhangyiqiu/sidechain-score-v1/foldingdiff/bc40_data_graph.pkl'
    transform = dataset.TorsionNoiseTransform()
    dsets = [dataset.ProteinDataset(cache = graph_data,
                                    split=s,
                                    #pickle_dir=full_data_name,
                                    transform=transform) for s in ('train', 'val')]

    effective_batch_size = batch_size
    if torch.cuda.is_available():
        effective_batch_size = int(batch_size / torch.cuda.device_count())
    pl.utilities.rank_zero_info(
        f"Given batch size: {batch_size} --> effective batch size with {torch.cuda.device_count()} GPUs: {effective_batch_size}"
    )

    datamodule = lightning.LightningDataset(train_dataset=dsets[0],
                                            val_dataset=dsets[1],
                                            batch_size=effective_batch_size,
                                            pin_memory=True)

    model = modelling.AngleDiffusion(
        lr=lr,
      # diffusion_fraction = 0.7,
        l2=l2_norm,
        l1=l1_norm,
        epochs=max_epochs,
        steps_per_epoch=len(datamodule.train_dataloader()),
        lr_scheduler=lr_scheduler,
        write_preds_to_dir=results_folder / "valid_preds" if write_valid_preds else None,
    )

    callbacks = build_callbacks(outdir=results_folder)

    # Get accelerator and distributed strategy
    accelerator, strategy = "cpu", None
    if not cpu_only and torch.cuda.is_available():
        accelerator = "cuda"
        if torch.cuda.device_count() > 1:
            # https://github.com/Lightning-AI/lightning/discussions/6761https://github.com/Lightning-AI/lightning/discussions/6761
            strategy = DDPStrategy(find_unused_parameters=False)

    logging.info(f"Using {accelerator} with strategy {strategy}")

    trainer = pl.Trainer(
        default_root_dir=results_folder,
        gradient_clip_val=gradient_clip,
        min_epochs=min_epochs,
        max_epochs=max_epochs,
        check_val_every_n_epoch=1,
        callbacks=callbacks,
        logger=pl.loggers.CSVLogger(save_dir=results_folder / "logs"),
        log_every_n_steps=len(datamodule.train_dataloader()),  # Log >= once per epoch
        accelerator=accelerator,
        strategy=strategy,
        gpus=ngpu,
        enable_progress_bar=False,
        move_metrics_to_cpu=False,  # Saves memory
    )

    torch.autograd.set_detect_anomaly(True)
    
    trainer.fit(
        model=model,
        datamodule=datamodule,
        #ckpt_path = '/mnt/petrelfs/zhangyiqiu/sidechain-score-v1/bin/result_122_crossIPA copy/models/best_by_valid/sample-mnist-epoch=226-mean_loss=0.541.ckpt'
    )

def build_parser() -> argparse.ArgumentParser:
    """
    Build CLI parser
    """
    parser = argparse.ArgumentParser(
        usage=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # https://stackoverflow.com/questions/4480075/argparse-optional-positional-arguments
    parser.add_argument(
        "config", nargs="?", default="", type=str, help="json of params"
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default=os.path.join(os.getcwd(), "results"),
        help="Directory to write model training outputs",
    )
    parser.add_argument(
        "--toy",
        type=int,
        default=None,
        help="Use a toy dataset of n items rather than full dataset",
    )
    parser.add_argument(
        "--debug_single_time",
        action="store_true",
        help="Debug single angle and timestep",
    )
    parser.add_argument("--cpu", action="store_true", help="Force use CPU")
    parser.add_argument(
        "--ngpu", type=int, default=-1, help="Number of GPUs to use (-1 for all)"
    )
    parser.add_argument("--dryrun", action="store_true", help="Dry run")
    parser.add_argument(
        "--layer",
        type=int,
        default=1,
        help="Use a toy dataset of n items rather than full dataset",
    )
    return parser


def main():
    """Run the training script based on params in the given json file"""
    parser = build_parser()
    args = parser.parse_args()

    # Load in parameters and run training loop
    config_args = {}  # Empty dictionary as default
    if args.config:
        with open(args.config) as source:
            config_args = json.load(source)
    config_args = utils.update_dict_nonnull(
        config_args,
        {
            "results_dir": args.outdir,
            "cpu_only": args.cpu,
            "ngpu": 6,
        },
    )    
    train(**config_args)


if __name__ == "__main__":
    curr_time = datetime.now().strftime("%y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(f"training_{curr_time}.log"),
            logging.StreamHandler(),
        ],
    )

    main()
