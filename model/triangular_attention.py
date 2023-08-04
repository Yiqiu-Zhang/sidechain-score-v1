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

from functools import partialmethod, partial
import math
from typing import Optional, List

import torch
import torch.nn as nn

from primitives import LayerNorm, Attention
from model.utils1 import permute_final_dims



class TriangleAttention(nn.Module):
    def __init__(
        self, c_z, c_hidden, no_heads, starting=True, inf=1e9
    ):
        """
        Args:
            c_z:
                Input channel dimension
            c_hidden:
                Overall hidden channel dimension (not per-head)
            no_heads:
                Number of attention heads
        """
        super(TriangleAttention, self).__init__()

        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.starting = starting
        self.inf = inf

        self.layer_norm = LayerNorm(self.c_z)

        self.linear = nn.Linear(self.c_z, self.no_heads, bias=False)

        self.mha = Attention(
            self.c_z, self.c_z, self.c_z, self.c_hidden, self.no_heads
        )


    def forward(self,
        x: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, N_rigid, N_rigid, c_z] input tensor (e.g. the pair representation)
        Returns:
            [*, N_rigid, N_rigid, c_z] output tensor
        """ 

        if not self.starting:
            x = x.transpose(-2, -3)
            mask = mask.transpose(-1, -2)

        # [*, N_rigid, N_rigid, c_z]
        x = self.layer_norm(x)

        # [*, N_rigid, 1, 1, N_rigid] 被 mask掉的 = -inf， 正常的 = 0
        mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]

        # [*, c_z, N_rigid, N_rigid]
        triangle_bias = permute_final_dims(self.linear(x), [2, 0, 1])

        # [*, 1, c_z, N_rigid, N_rigid]
        triangle_bias = triangle_bias.unsqueeze(-4)

        biases = [mask_bias, triangle_bias]

        # [*, N_rigid, N_rigid, c_z]
        x = self.mha(q_x=x, kv_x=x, biases=biases)

        if not self.starting:
            x = x.transpose(-2, -3)

        return x


# Implements Algorithm 13
TriangleAttentionStartingNode = TriangleAttention


class TriangleAttentionEndingNode(TriangleAttention):
    """
    Implements Algorithm 14.
    """
    __init__ = partialmethod(TriangleAttention.__init__, starting=False)
