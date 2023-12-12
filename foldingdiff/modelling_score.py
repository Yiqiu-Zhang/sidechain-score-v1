"""
Modelling
"""
import os
import shutil
import time
import glob
from pathlib import Path
import json
import logging
from typing import *

import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl

from transformers.optimization import get_linear_schedule_with_warmup

from tqdm.auto import tqdm

from foldingdiff import losses_score as losses


import sys
sys.path.append(r"/mnt/petrelfs/lvying/code/sidechain_score/")
from model import *
from model.rigid_diffusion_score import *

from torch.nn.parallel import DistributedDataParallel

LR_SCHEDULE = Optional[Literal["OneCycleLR", "LinearWarmup"]]
TIME_ENCODING = Literal["gaussian_fourier", "sinusoidal"]
LOSS_KEYS = Literal["l1", "smooth_l1"]
DECODER_HEAD = Literal["mlp", "linear"]

class AngleDiffusionBase(nn.Module):
    """
    Our Model
    """

    def __init__(self) -> None:

        """
        dim should be the dimension of the inputs
        """
        super().__init__()  

        self.encoder = RigidDiffusion()

        self.train_epoch_counter = 0
        self.train_epoch_last_time = time.time()

    @classmethod
    def from_dir(
        cls,
        dirname: str,
        load_weights: bool = True,
        best_by: Literal["train", "valid"] = "valid",
        copy_to: str = "",
        **kwargs,
    ):
        """
        Builds this model out from directory. Legacy mode is for loading models
        before there were separate folders for training and validation best models.
        idx indicates which model to load if multiple are given
        """
        train_args_fname = os.path.join(dirname, "training_args.json")
        with open(train_args_fname, "r") as source:
            train_args = json.load(source)

        if load_weights:
            subfolder = f"best_by_{best_by}"
            ckpt_names = glob.glob(os.path.join(dirname, "models", subfolder, "*.ckpt"))
            logging.info(f"Found {len(ckpt_names)} checkpoints")
            ckpt_name = ckpt_names[0] # choose which check point
            logging.info(f"Loading weights from {ckpt_name}")

            retval = cls()
            loaded = torch.load(ckpt_name, map_location=torch.device("cuda"))
            retval.load_state_dict(loaded["state_dict"])

        # If specified, copy out the requisite files to the given directory
        if copy_to:
            logging.info(f"Copying minimal model file set to: {copy_to}")
            os.makedirs(copy_to, exist_ok=True)
            copy_to = Path(copy_to)
            with open(copy_to / "training_args.json", "w") as sink:
                json.dump(train_args, sink)
            if load_weights:
                # Create the direcotry structure
                ckpt_dir = copy_to / "models" / subfolder
                os.makedirs(ckpt_dir, exist_ok=True)
                shutil.copyfile(ckpt_name, ckpt_dir / os.path.basename(ckpt_name))

        return retval

    def forward(self,data):

        score = self.encoder(data)

        return score#, local_trans

class AngleDiffusion(AngleDiffusionBase, pl.LightningModule):
    """
    Wraps our model as a pl LightningModule for easy training
    """

    def __init__(
        self,
        lr: float = 5e-5,
        l2: float = 0.0,
        l1: float = 0.0,
        circle_reg: float = 0.0,
        epochs: int = 10,
        steps_per_epoch: int = 250,  # Dummy value
        #diffusion_fraction: float = 0.7,
        lr_scheduler: LR_SCHEDULE = None,
        write_preds_to_dir: Optional[str] = None,
        **kwargs,
        
    ):
        """Feed args to BertForDiffusionBase and then feed the rest into"""
        AngleDiffusionBase.__init__(self, **kwargs)
        # Store information about leraning rates and loss
        #self.diffusion_fraction = diffusion_fraction
        self.learning_rate = lr

        self.l1_lambda = l1
        self.l2_lambda = l2
        self.circle_lambda = circle_reg
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.lr_scheduler = lr_scheduler
        
        # Set up the output directory for writing predictions
        self.write_preds_to_dir = write_preds_to_dir
        self.write_preds_counter = 0
        if self.write_preds_to_dir:
            os.makedirs(self.write_preds_to_dir, exist_ok=True)
  
    def training_step(self, batch, batch_idx):
        """
        Training step, runs once per batch
        """
        
        predicted_score = self.forward(batch)
        avg_loss = losses.score_loss(predicted_score, batch)

        self.log("train_loss", avg_loss, on_epoch=True,batch_size=batch.batch_size, rank_zero_only=True)

        return avg_loss

    def training_epoch_end(self, outputs) -> None:

        """Log the average training loss over the epoch"""
        t_delta = time.time() - self.train_epoch_last_time
        pl.utilities.rank_zero_info(
            f"Time at epoch {self.train_epoch_counter} end: ({t_delta:.2f} seconds)"
        )
        # Increment counter and timers
        self.train_epoch_counter += 1
        self.train_epoch_last_time = time.time()

    def validation_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        """
        Validation step
        """
        with torch.no_grad():
            predicted_score = self.forward(batch)
            avg_loss = losses.score_loss(predicted_score, batch)
            
        self.log("val_loss", avg_loss, on_epoch=True, batch_size=batch.batch_size, rank_zero_only=True)
        return avg_loss

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Return optimizer. Limited support for some optimizers
        """
        optim = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_lambda,
        )
        retval = {"optimizer": optim}
        if self.lr_scheduler:
            if self.lr_scheduler == "OneCycleLR":
                retval["lr_scheduler"] = {
                    "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                        optim,
                        max_lr=1e-2,
                        epochs=self.epochs,
                        steps_per_epoch=self.steps_per_epoch,
                    ),
                    "monitor": "val_loss",
                    "frequency": 1,
                    "interval": "step",
                }
            elif self.lr_scheduler == "LinearWarmup":
                # https://huggingface.co/docs/transformers/v4.21.2/en/main_classes/optimizer_schedules#transformers.get_linear_schedule_with_warmup
                # Transformers typically do well with linear warmup
                warmup_steps = int(self.epochs * 0.1)
                pl.utilities.rank_zero_info(
                    f"Using linear warmup with {warmup_steps}/{self.epochs} warmup steps"
                )
                retval["lr_scheduler"] = {
                    "scheduler": get_linear_schedule_with_warmup(
                        optim,
                        num_warmup_steps=warmup_steps,
                        num_training_steps=self.epochs,
                    ),
                    "frequency": 1,
                    "interval": "epoch",  # Call after 1 epoch
                }
            else:
                raise ValueError(f"Unknown lr scheduler {self.lr_scheduler}")
        pl.utilities.rank_zero_info(f"Using optimizer {retval}")
        return retval
    
    def configure_ddp(self, model, device_ids):
        ddp = DistributedDataParallel(
            model,
            device_ids=device_ids,
            find_unused_parameters=True  # 设置find_unused_parameters=True
        )
        return ddp