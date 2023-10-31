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


class GaussianFourierProjection(nn.Module):
    """
    Gaussian random features for encoding time steps.
    Built primarily for score-based models.

    Source:
    https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing#scrollTo=YyQtV7155Nht
    """

    def __init__(self, embed_dim: int, scale: float = 2 * torch.pi):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        w = torch.randn(embed_dim // 2) * scale
        assert w.requires_grad == False
        self.register_buffer("W", w)

    def forward(self, x: torch.Tensor):
        """
        takes as input the time vector and returns the time encoding
        time (x): (batch_size, )
        output  : (batch_size, embed_dim)
        """
        if x.ndim > 1:
            x = x.squeeze()
        elif x.ndim < 1:
            x = x.unsqueeze(0)
        x_proj = x[:, None] * self.W[None, :] * 2 * torch.pi
        embed = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return embed


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Positional embeddings
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        # half_dim shape
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        # outer product (batch, 1) x (1, half_dim) -> (batch x half_dim)
        embeddings = time[:, None] * embeddings[None, :]
        # sin and cosine embeddings
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class PositionalEncoding(nn.Module):
    """
    Positional embedding for BERT.
    Source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        assert len(x.shape) == 3
        orig_shape = x.shape
        # x is a tensor of shape (batch_size, seq_len, embedding_dim)
        # permute to be (seq_len, batch_size, embedding_dim)
        x = x.permute(1, 0, 2)
        x += self.pe[: x.size(0)]
        # permute back to (batch_size, seq_len, embedding_dim)
        x = x.permute(1, 0, 2)
        assert x.shape == orig_shape, f"{x.shape} != {orig_shape}"
        return self.dropout(x)


class BertEmbeddings(nn.Module):
    """
    Adds in positional embeddings if using absolute embeddings, adds layer norm and dropout
    """

    def __init__(self, config):
        super().__init__()
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "absolute":
            self.position_embeddings = nn.Embedding(
                config.max_position_embeddings, config.hidden_size
            )
            self.register_buffer(
                "position_ids",
                torch.arange(config.max_position_embeddings).expand((1, -1)),
            )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized

    def forward(
        self,
        input_embeds: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> torch.Tensor:
        assert position_ids is not None, "`position_ids` must be defined"
        embeddings = input_embeds
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class AnglesPredictor(nn.Module):
    """
    Predict angles from the embeddings. For BERT, the MLM task is done using an
    architecture like
    d_model -> dense -> d_model -> activation -> layernorm -> dense -> d_output
    https://github.com/huggingface/transformers/blob/v4.21.1/src/transformers/models/bert/modeling_bert.py#L681

    activation should be given as nn.ReLU for example -- NOT nn.ReLU()
    """

    def __init__(
        self,
        d_model: int,
        d_out: int = 4,
        activation: Union[str, nn.Module] = "gelu",
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_out = d_out
        self.dense1 = nn.Linear(d_model, d_model)

        if isinstance(activation, str):
            self.dense1_act = get_activation(activation)
        else:
            self.dense1_act = activation()
        self.layer_norm = nn.LayerNorm(d_model, eps=eps)

        self.dense2 = nn.Linear(d_model, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense1(x)
        x = self.dense1_act(x)
        x = self.layer_norm(x)
        x = self.dense2(x)
        return x


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
        if ft_names is not None:
            self.ft_names = ft_names
        else:
            self.ft_names = [f"ft{i}" for i in range(n_inputs)]
        assert (
            len(self.ft_names) == n_inputs
        ), f"Got {len(self.ft_names)} names, expected {n_inputs}"

        self.encoder = RigidDiffusion()
        
        # Set up the time embedder
      #  if time_encoding == "gaussian_fourier":
      #      self.time_embed = GaussianFourierProjection(d_model)
      #  elif time_encoding == "sinusoidal":
      #      self.time_embed = SinusoidalPositionEmbeddings(d_model)
      #  else:
      #      raise ValueError(f"Unknown time encoding: {time_encoding}")
      #  pl.utilities.rank_zero_info(f"Using time embedding: {self.time_embed}")
        
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

    def forward(
        self,
        rigids,
        #local_r,
        seq_idx: torch.Tensor,#[batch,128,4]
       # diffusion_mask: torch.Tensor, #[batch,128,4]
        timestep: torch.Tensor, 
        x_seq_esm: torch.Tensor,  #[batch,128,1024]
        x_rigid_type: torch.Tensor, #[batch,128,5,20] x_rigid_type[-1]=one hot
        x_rigid_property: torch.Tensor, #[batch,128,5,6]
        pad_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        score = self.encoder(rigids, #, local_trans
                            seq_idx,
                            timestep,
                            x_seq_esm,
                            x_rigid_type,
                            x_rigid_property,
                            pad_mask,
        )

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
                if is_angular
                else self.nonangular_loss_fn_dict[loss]
                for is_angular in self.ft_is_angular
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
                             batch: torch.Tensor,
                             ) -> torch.Tensor:
        """
           Returns the loss terms for the model. Length of the returned list
           is equivalent to the number of features we are fitting to.
        """

        # [*, N_res] Rigid

        # [*, N_rigid] Rigid
        corrupted_sin_cos = torch.stack(
            [torch.sin(batch['corrupted']), torch.cos(batch['corrupted'])], 
            dim=-1)
        rigids, _, _ = structure_build.torsion_to_frame(corrupted_sin_cos,
                                                        batch["seq"],
                                                        batch["coords"])


        predicted_score = self.forward(rigids, #, local_trans
                                                    batch["seq"],  # [batch,128,4]
                                                    #diffusion_mask,  # [batch,128,1]
                                                    batch["t"],
                                                    batch["acid_embedding"],  # [batch,128,1024]
                                                    batch['rigid_type_onehot'],  # [batch,128,5,19] x_rigid_type[-1]=one hot
                                                    batch['rigid_property'],  # [batch,128,5,6]
                                                    batch["attn_mask"],
                                                )

        loss_fn = self.angular_loss_fn_dict['score_loss']

        loss = loss_fn(
            predicted_score,
            #local_trans,
            batch["known_noise"],
            batch["t"], # sigma
            batch['seq'],  # [b,L] restpyes in number
            batch['angles'],
            batch["coords"],
            rigids,
            batch['chi_mask'],
            all_loc = self.pred_all    # [b,L,4]  Padded in chi_mask so no need to use padding mask
        )

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

class BertForAutoregressiveBase(AngleDiffusionBase):
    """
    Overrides the previous model's forward function to not handle noise or timesteps
    """

    def forward(
        self,
        inputs: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_lengths: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        assert len(inputs.shape) == 3  # batch_size, seq_length, features
        inputs_upscaled = self.inputs_to_hidden_dim(inputs)  # Batch * seq_len * dim

        # Embed the lengths - note that we are reusing the time embedding here
        # Shape (batch, embed) -> (batch, 1, embed)
        len_embed = self.time_embed(seq_lengths).unsqueeze(1)
        inputs_upscaled += len_embed

        if position_ids is None:
            batch_size, seq_length, *_ = inputs.size()
            # Shape (batch, seq_len)
            position_ids = (
                torch.arange(
                    seq_length,
                )
                .expand(batch_size, -1)
                .to(inputs.device)
            )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads. This code is taken
        # from hugggingface modeling_utils
        assert (
            attention_mask.dim() == 2
        ), f"Attention mask expected in shape (batch_size, seq_length), got {attention_mask.shape}"
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = extended_attention_mask.type_as(attention_mask)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        inputs_upscaled = self.embeddings(inputs_upscaled, position_ids=position_ids)
        encoder_outputs = self.encoder(
            inputs_upscaled,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        per_token_decoded = self.token_decoder(sequence_output)
        return per_token_decoded

    @torch.no_grad()
    def sample(
        self,
        seed_angles: torch.Tensor,
        seq_lengths: torch.Tensor,
        num_seed: int = 2,
        pbar: bool = True,
    ) -> List[torch.Tensor]:
        """
        Sample a set of angles of seq_lengths given a series of seed angles
        seed_angles should be given as a tensor of (batch, seq_len, num_angles)
        The first num_seed angles are taken as fixed and the rest are autoregressively
        generated
        """
        assert torch.all(seed_angles[:, :num_seed, :] <= torch.pi)
        assert torch.all(seed_angles[:, :num_seed, :] >= -torch.pi)
        retval = seed_angles.clone().to(seed_angles.device)
        assert seed_angles.ndim == 3

        attention_mask = torch.zeros(seed_angles.shape[:2]).to(seed_angles.device)
        for i in tqdm(range(num_seed, torch.max(seq_lengths).item()), disable=not pbar):
            attention_mask[:, :i] = 1.0
            assert torch.all(attention_mask.sum(axis=1) == i)
            next_angle = self.forward(
                retval,
                attention_mask=attention_mask,
                seq_lengths=seq_lengths,
            )[:, i, :]
            retval[:, i, :] = next_angle
        return [retval[i, :l, :] for i, l in enumerate(seq_lengths)]


class BertForAutoregressive(BertForAutoregressiveBase, pl.LightningModule):
    """
    Wraps model in a pl.LightningModule for easy training as an
    autoregressive model where we are interested in predicting the next set of
    angles given the current set of angles
    """

    def __init__(
        self,
        loss_key: LOSS_KEYS = "smooth_l1",
        lr: float = 5e-5,
        lr_scheduler: Optional[str] = None,
        l2: float = 0.0,
        epochs: int = 1,
        steps_per_epoch: int = 250,  # Dummy value
        **kwargs,
    ):
        AngleDiffusionBase.__init__(self, **kwargs)
        self.learning_rate = lr
        self.lr_scheduler = lr_scheduler
        self.l2_lambda = l2
        self.loss = self.angular_loss_fn_dict[loss_key]
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch

    def _get_loss(self, batch) -> torch.Tensor:
        """
        Get the loss terms for a batch
        """
        # Get the predictions
       
        
        preds = self.forward(
            batch["angles"],
            attention_mask=batch["causal_attn_mask"],
            seq_lengths=batch["lengths"],
            position_ids=batch["position_ids"],
        )
        assert preds.ndim == 3  # batch_size, seq_length, features
        # Get the loss terms
        l = self.loss(
            preds[torch.arange(batch["lengths"].shape[0]), batch["causal_idx"]],
            batch["causal_target"],
        )
       
        return l

    def training_step(self, batch, batch_idx):
        loss = self._get_loss(batch)
        self.log("train_loss", loss, rank_zero_only=True)
        return loss

    def training_epoch_end(self, outputs) -> None:
        """Log average training loss over epoch"""
        losses = torch.stack([o["loss"] for o in outputs])
        mean_loss = torch.mean(losses)
        t_delta = time.time() - self.train_epoch_last_time
        pl.utilities.rank_zero_info(
            f"Train loss at epoch {self.train_epoch_counter} end: {mean_loss:.4f} ({t_delta:.2f} seconds)"
        )
        # Increment counter and timers
        self.train_epoch_counter += 1
        self.train_epoch_last_time = time.time()

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            loss = self._get_loss(batch)
        self.log("val_loss", loss, rank_zero_only=True)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs) -> None:
        losses = torch.stack([o["val_loss"] for o in outputs])
        mean_loss = torch.mean(losses)
        pl.utilities.rank_zero_info(
            f"Valid loss at epoch {self.train_epoch_counter} end: {mean_loss:.4f}"
        )

    def configure_optimizers(self) -> Dict[str, Any]:
        optim = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.l2_lambda
        )
        retval = {"optimizer": optim}
        pl.utilities.rank_zero_info(f"Using optimizer {retval}")

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

        return retval


def main():
    """on the fly testing"""
    m = BertForAutoregressiveBase.from_dir(
        "/home/wukevin/projects/protdiff_results/models/ar_baseline/results"
    )
    # rand samples uniformly from [0, 1) so we expand the range and shift it
    rand_angles = torch.rand(size=(32, 128, 6)) * 2 * torch.pi - torch.pi
    rand_lens = torch.randint(low=40, high=128, size=(32,))
    m.sample(seed_angles=rand_angles, seq_lengths=rand_lens)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()