"""
Modelling
"""
import os
import random
import re
import shutil
import time
import glob
from pathlib import Path
import json
import inspect
import logging
import math
import functools
from typing import *

import torch
from torch import nn
from torch.nn import functional as F

import pytorch_lightning as pl

from transformers import BertConfig
from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel,
    BertEncoder,
)
from transformers.activations import get_activation
from transformers.optimization import get_linear_schedule_with_warmup

from tqdm.auto import tqdm

from foldingdiff import losses_score as losses
from foldingdiff import nerf
from foldingdiff.datasets_score import FEATURE_SET_NAMES_TO_ANGULARITY

#start===================yinglv====================================
# import graph_transformer
#from rigid_attention.rigid_angle_transformer import *
from write_preds_pdb import structure_build_score as structure_build
from write_preds_pdb import geometry
import sys
sys.path.append(r"/mnt/petrelfs/lvying/code/sidechain_score/")
from model import *
from model.rigid_diffusion_score import *
#end=====================yinglv====================================
import tracemalloc
from torch.nn.parallel import DistributedDataParallel

LR_SCHEDULE = Optional[Literal["OneCycleLR", "LinearWarmup"]]
TIME_ENCODING = Literal["gaussian_fourier", "sinusoidal"]
LOSS_KEYS = Literal["l1", "smooth_l1"]
DECODER_HEAD = Literal["mlp", "linear"]

class AngleDiffusionBase(nn.Module):
    """
    Our Model
    """

    # Define loss functions and their wrapped angular versions
    nonangular_loss_fn_dict = {
        "l1": F.l1_loss,
        "smooth_l1": F.smooth_l1_loss,
    }
    angular_loss_fn_dict = {
        "l1": losses.radian_l1_loss,
        "smooth_l1": functools.partial(
            losses.radian_smooth_l1_loss, beta=torch.pi / 10
        ),
        #========================new loss==============================
        "sin_cos": losses.square_chi_loss,
        "square_chi_loss_with_periodic": losses.square_chi_loss_with_periodic,
        "score_loss": losses.score_loss
        #========================new loss==============================
    }
    # To have legacy models still work with these
    loss_autocorrect_dict = {
        "radian_l1_smooth": "smooth_l1",
    }

    def __init__(
        self,
        config,
        ft_is_angular: List[bool] = [True, True, True, True],
        ft_names: Optional[List[str]] = None,
        time_encoding: TIME_ENCODING = "gaussian_fourier",
        d_model: int = 384,
        d_esm_seq: int = 320,
        n_rigid_type: int = 20,
        n_rigid_property: int = 6,
        n_layers: int = 4,
        n_heads: int = 8,
        d_ff: int = 1024, #hidden layer dim
        d_angles:int = 4,
        max_seq_len: int = 5000,
        dropout: float = 0.1,
        pred_all: bool = False,
    ) -> None:
        """
        dim should be the dimension of the inputs
        """
        super().__init__()  
        self.pred_all = pred_all
        self.config = config
        self.ft_is_angular = ft_is_angular
        n_inputs = len(ft_is_angular)
        self.n_inputs = n_inputs

        self.encoder = RigidDiffusion()

        # Epoch counters and timers
        self.train_epoch_counter = 0
        self.train_epoch_last_time = time.time()

    @classmethod
    def from_dir(
        cls,
        dirname: str,
        ft_is_angular: Optional[Sequence[bool]] = None,
        load_weights: bool = True,
        idx: int = -1,
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
        config = BertConfig.from_json_file(os.path.join(dirname, "config.json"))

        if ft_is_angular is None:
            ft_is_angular = FEATURE_SET_NAMES_TO_ANGULARITY[
                train_args["angles_definitions"]
            ]
            logging.info(f"Auto constructed ft_is_angular: {ft_is_angular}")

        # Handles the case where we repurpose the time encoding for seq len encoding in the AR model
        time_encoding_key = (
            "time_encoding" if "time_encoding" in train_args else "seq_len_encoding"
        )
        model_args = dict(
            config=config,
            ft_is_angular=ft_is_angular,
            time_encoding=train_args[time_encoding_key],
            #dropout = 0.1
            # lr=train_args["lr"],
            # loss=train_args["loss"],
            # l2=train_args["l2_norm"],
            # l1=train_args["l1_norm"],
            # circle_reg=train_args["circle_reg"],
            # lr_scheduler=train_args["lr_scheduler"],
            **kwargs,
        )
        print('==============================model_args', model_args)
        if load_weights:
            subfolder = f"best_by_{best_by}"
            ckpt_names = glob.glob(os.path.join(dirname, "models", subfolder, "*.ckpt"))
            logging.info(f"Found {len(ckpt_names)} checkpoints")
            ckpt_name = ckpt_names[idx]
            logging.info(f"Loading weights from {ckpt_name}")
            if hasattr(cls, "load_from_checkpoint"):
                # Defined for pytorch lightning module
                retval = cls.load_from_checkpoint(
                    checkpoint_path=ckpt_name, **model_args
                )
            else:
                retval = cls(**model_args)
                loaded = torch.load(ckpt_name, map_location=torch.device("cpu"))
                retval.load_state_dict(loaded["state_dict"])
        else:
            retval = cls(**model_args)
            logging.info(f"Loaded unitialized model from {dirname}")

        # If specified, copy out the requisite files to the given directory
        if copy_to:
            logging.info(f"Copying minimal model file set to: {copy_to}")
            os.makedirs(copy_to, exist_ok=True)
            copy_to = Path(copy_to)
            with open(copy_to / "training_args.json", "w") as sink:
                json.dump(train_args, sink)
            config.save_pretrained(copy_to)
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
        loss: Union[Callable, LOSS_KEYS] = "square_chi_loss_with_periodic", # loss is str or function
        use_pairwise_dist_loss: Union[float, Tuple[float, float, int]] = 0.0,
        l2: float = 0.0,
        l1: float = 0.0,
        circle_reg: float = 0.0,
        epochs: int = 10,
        steps_per_epoch: int = 250,  # Dummy value
        diffusion_fraction: float = 0.7,
        lr_scheduler: LR_SCHEDULE = None,
        write_preds_to_dir: Optional[str] = None,
        **kwargs,
        
    ):
        """Feed args to BertForDiffusionBase and then feed the rest into"""
        AngleDiffusionBase.__init__(self, **kwargs)
        # Store information about leraning rates and loss
        self.diffusion_fraction = diffusion_fraction
        self.learning_rate = lr
        # loss function is either a callable or a list of callables
        if isinstance(loss, str):
            logging.info(
                f"Mapping loss {loss} to list of losses corresponding to angular {self.ft_is_angular}"
            )
            if loss in self.loss_autocorrect_dict:
                logging.info(
                    "Autocorrecting {} to {}".format(
                        loss, self.loss_autocorrect_dict[loss]
                    )
                )
                loss = self.loss_autocorrect_dict[loss]  # 等号左侧的loss为str，右侧的也为str
            self.loss_func = [
                self.angular_loss_fn_dict[loss]

            ]
        else:
            logging.warning(
                f"Using pre-given callable loss: {loss}. This may not handle angles correctly!"
            )
            self.loss_func = loss
        pl.utilities.rank_zero_info(f"Using loss: {self.loss_func}")
        if isinstance(self.loss_func, (tuple, list)):
            assert (
                len(self.loss_func) == self.n_inputs
            ), f"Got {len(self.loss_func)} loss functions, expected {self.n_inputs}"

        self.use_pairwise_dist_loss = use_pairwise_dist_loss
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
  
    def _get_loss_terms_grad(self,
                             data: torch.Tensor,
                             ) -> torch.Tensor:
        """
           Returns the loss terms for the model. Length of the returned list
           is equivalent to the number of features we are fitting to.
        """

        predicted_score = self.forward(data)

        loss_fn = self.angular_loss_fn_dict['score_loss']

        loss = loss_fn(predicted_score, data)

        return loss

    def training_step(self, batch, batch_idx):
        """
        Training step, runs once per batch
        """
       # torch.autograd.set_detect_anomaly(True)

        avg_loss = self._get_loss_terms_grad(batch)
        
        self.log("train_loss", avg_loss, on_epoch=True)

        return avg_loss

    def training_epoch_end(self, outputs) -> None:
      #  tracemalloc.start()
        """Log the average training loss over the epoch"""
        losses = torch.stack([o["loss"] for o in outputs])
        mean_loss = torch.mean(losses)
        t_delta = time.time() - self.train_epoch_last_time
        pl.utilities.rank_zero_info(
            f"Train loss at epoch {self.train_epoch_counter} end: {mean_loss:.4f} ({t_delta:.2f} seconds)"
        )
        # Increment counter and timers
        #print("============hparams==============", self.hparams)
        self.train_epoch_counter += 1
        self.train_epoch_last_time = time.time()

    def validation_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        """
        Validation step
        """
        with torch.no_grad():
            avg_loss = self._get_loss_terms_grad(batch)
            self.write_preds_counter += 1

        self.log("val_loss", avg_loss, on_epoch=True)
        return avg_loss

    def validation_epoch_end(self, outputs) -> None:
        """Log the average validation loss over the epoch"""
        # Note that this method is called before zstraining_epoch_end().
        losses = torch.stack([o for o in outputs])
        mean_loss = torch.mean(losses)
        
        pl.utilities.rank_zero_info(
            f"Valid loss at epoch {self.train_epoch_counter} end: {mean_loss:.4f}"
        )

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