# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Optional, Callable, List, Tuple
import numpy as np

import torch
import torch.nn as nn
from scipy.stats import truncnorm

from model.utils1 import permute_final_dims, flatten_final_dims

DEFAULT_LMA_Q_CHUNK_SIZE=1024
DEFAULT_LMA_KV_CHUNK_SIZE=4096


def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f


def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights):
    nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def normal_init_(weights):
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")


def ipa_point_weights_init_(weights):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)

'''
class Linear(nn.Linear):
    """
    A Linear layer with built-in nonstandard initializations. Called just
    like torch.nn.Linear.

    Implements the initializers in 1.11.4, plus some additional ones found
    in the code.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        init: str = "default",
        init_fn: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
    ):
        """
        Args:
            in_dim:
                The final dimension of inputs to the layer
            out_dim:
                The final dimension of layer outputs
            bias:
                Whether to learn an additive bias. True by default
            init:
                The initializer to use. Choose from:

                "default": LeCun fan-in truncated normal initialization
                "relu": He initialization w/ truncated normal distribution
                "glorot": Fan-average Glorot uniform initialization
                "gating": Weights=0, Bias=1
                "normal": Normal initialization with std=1/sqrt(fan_in)
                "final": Weights=0, Bias=0

                Overridden by init_fn if the latter is not None.
            init_fn:
                A custom initializer taking weight and bias as inputs.
                Overrides init if not None.
        """
        super(Linear, self).__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        with torch.no_grad():
            if init_fn is not None:
                init_fn(self.weight, self.bias)
            else:
                if init == "default":
                    lecun_normal_init_(self.weight)
                elif init == "relu":
                    he_normal_init_(self.weight)
                elif init == "glorot":
                    glorot_uniform_init_(self.weight)
                elif init == "gating":
                    gating_init_(self.weight)
                    if bias:
                        self.bias.fill_(1.0)
                elif init == "normal":
                    normal_init_(self.weight)
                elif init == "final":
                    final_init_(self.weight)
                else:
                    raise ValueError("Invalid init string.")
'''

class LayerNorm(nn.Module):
    def __init__(self, c_in, eps=1e-5):
        super(LayerNorm, self).__init__()

        self.c_in = (c_in,)
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(c_in))
        self.bias = nn.Parameter(torch.zeros(c_in))

    def forward(self, x):
        d = x.dtype

        out = nn.functional.layer_norm(
            x,
            self.c_in,
            self.weight,
            self.bias,
            self.eps,
        )

        return out


@torch.jit.ignore
def softmax_no_cast(t: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
        Softmax, but without automatic casting to fp32 when the input is of
        type bfloat16
    """
    d = t.dtype

    s = torch.nn.functional.softmax(t, dim=dim)

    return s


#@torch.jit.script
def _attention(query: torch.Tensor, # [*, N_rigid, H, N_rigid, C_hidden]
               key: torch.Tensor,
               value: torch.Tensor,
               biases: List[torch.Tensor] # [[*, N_rigid, 1, 1, N_rigid],
                                          #  [*, 1, c_z, N_rigid, N_rigid]]
               ) -> torch.Tensor:

    # [*, H, C_hidden, K]
    # [*, N_rigid, H, C_hidden, N_rigid]
    key = permute_final_dims(key, [1, 0])

    # [*, H, Q, K]
    # [*, N_rigid, H, N_rigid, N_rigid]
    a = torch.matmul(query, key)
    #print("==========a===========",a.device)
    for b in biases:
     #   print("==========b===========",b.device)
        
        a += b.to(a.device)

    a = softmax_no_cast(a, -1)

    # [*, H, Q, C_hidden]
    # [*, N_rigid, H, N_rigid, C_hidden]
    a = torch.matmul(a, value)

    return a


class Attention(nn.Module):
    """
    Standard multi-head attention using AlphaFold's default layer
    initialization. Allows multiple bias vectors.
    """
    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        no_heads: int,
        gating: bool = True,
    ):
        """
        Args:
            c_q:
                Input dimension of query data
            c_k:
                Input dimension of key data
            c_v:
                Input dimension of value data
            c_hidden:
                Per-head hidden dimension
            no_heads:
                Number of attention heads
            gating:
                Whether the output should be gated using query data
        """
        super(Attention, self).__init__()

        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating

        # DISCREPANCY: c_hidden is not the per-head channel dimension, as
        # stated in the supplement, but the overall channel dimension.

        self.linear_q = nn.Linear(
            self.c_q, self.c_hidden * self.no_heads, bias=False
        )
        self.linear_k = nn.Linear(
            self.c_k, self.c_hidden * self.no_heads, bias=False
        )
        self.linear_v = nn.Linear(
            self.c_v, self.c_hidden * self.no_heads, bias=False
        )
        self.linear_o = nn.Linear(
            self.c_hidden * self.no_heads, self.c_q
        )

        self.linear_g = None
        if self.gating:
            self.linear_g = nn.Linear(
                self.c_q, self.c_hidden * self.no_heads
            )

        self.sigmoid = nn.Sigmoid()

    def _prep_qkv(self,
        q_x: torch.Tensor,  # [*, N_rigid, N_rigid, c_z]
        kv_x: torch.Tensor  # [*, N_rigid, N_rigid, c_z]
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        # [*, N_rigid, N_rigid, H * C_hidden]
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        # [*, N_rigid, N_rigid, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        # [*, H, Q/K, C_hidden] [*, N_rigid, H, N_rigid, C_hidden]
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        q /= math.sqrt(self.c_hidden)

        return q, k, v

    def _wrap_up(self,
        o: torch.Tensor, # [*, N_rigid, N_rigid, H, C_hidden]
        q_x: torch.Tensor # [*, N_rigid, N_rigid, c_z]
    ) -> torch.Tensor:
        if self.linear_g is not None:
            # [*, N_rigid, N_rigid, H*C_hidden]
            g = self.sigmoid(self.linear_g(q_x))
        
            # [*, Q, H, C_hidden]
            # [*, N_rigid, N_rigid, H, C_hidden]
            g = g.view(g.shape[:-1] + (self.no_heads, -1))
            o = o * g

        # [*, Q, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, Q, C_q]
        o = self.linear_o(o)

        return o

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        biases: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            q_x:
                [*, Q, C_q] query data
            kv_x:
                [*, K, C_k] key data
            biases:
                List of biases that broadcast to [*, H, Q, K]
        Returns
            [*, Q, C_q] attention update
        """

        if biases is None:
            biases = []
        
        # [*, N_rigid, H, N_rigid, C_hidden]
        q, k, v = self._prep_qkv(q_x, kv_x)

        # [*, Q, H, C_hidden]
        o = _attention(q, k, v, biases) # [*, N_rigid, H, N_rigid, C_hidden]
        o = o.transpose(-2, -3)# [*, N_rigid, N_rigid, H, C_hidden]

        # [*, N_rigid, N_rigid, c_z]
        o = self._wrap_up(o, q_x)

        return o