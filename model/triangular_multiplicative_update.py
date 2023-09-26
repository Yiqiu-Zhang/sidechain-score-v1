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

from functools import partialmethod
from typing import Optional

import torch
import torch.nn as nn

from model.primitives import LayerNorm
from model.utils1 import permute_final_dims
from model.precision_utilis import is_fp16_enabled

class TriangleMultiplicativeUpdate(nn.Module):
    """
    Implements Algorithms 11 and 12.
    """
    def __init__(self, c_z, c_hidden, _outgoing=True):
        """
        Args:
            c_z:
                Input channel dimension
            c:
                Hidden channel dimension
        """
        super(TriangleMultiplicativeUpdate, self).__init__()
        self.c_z = c_z
        self.c_hidden = c_hidden
        self._outgoing = _outgoing

        self.linear_a_p = nn.Linear(self.c_z, self.c_hidden)
        self.linear_a_g = nn.Linear(self.c_z, self.c_hidden)
        self.linear_b_p = nn.Linear(self.c_z, self.c_hidden)
        self.linear_b_g = nn.Linear(self.c_z, self.c_hidden)
        self.linear_g = nn.Linear(self.c_z, self.c_z)
        self.linear_z = nn.Linear(self.c_hidden, self.c_z)

        self.layer_norm_in = LayerNorm(self.c_z)
        self.layer_norm_out = LayerNorm(self.c_hidden)

        self.sigmoid = nn.Sigmoid()

    def _combine_projections(self,
        a: torch.Tensor,
        b: torch.Tensor,
        _inplace_chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        if self._outgoing:
            a = permute_final_dims(a, [2, 0, 1])
            b = permute_final_dims(b, [2, 1, 0])
        else:
            a = permute_final_dims(a, [2, 1, 0]) # [*, C_hidden, N_rigid(1), N_rigid(0)]
            b = permute_final_dims(b,  [2, 0, 1])# [*, C_hidden, N_rigid, N_rigid]

        p = torch.matmul(a, b)

        return permute_final_dims(p, [1, 2, 0]) # [*, N_rigid, N_rigid, C_hidden]

    def forward(self, 
        z: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            z:
                [*, N_rigid, N_rigid, C_z] input tensor
            mask:
                [*, N_rigid, N_rigid] input mask
        Returns:
            [*, N_rigid, N_rigid, C_z] output tensor
        """

        mask = mask.unsqueeze(-1)
        
        z = self.layer_norm_in(z)
        a = mask
        a = a.to('cuda') * self.sigmoid(self.linear_a_g(z)) 
        a = a.to('cuda') * self.linear_a_p(z)
        b = mask
        b = b.to('cuda') * self.sigmoid(self.linear_b_g(z))
        b = b.to('cuda') * self.linear_b_p(z)
        
        if is_fp16_enabled():
            with torch.cuda.amp.autocast(enabled=False):
                x = self._combine_projections(a.float(), b.float())
        else:
            x = self._combine_projections(a, b)
        
        del a, b
        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        g = self.sigmoid(self.linear_g(z))
        x = x * g

        return x


class TriangleMultiplicationOutgoing(TriangleMultiplicativeUpdate):
    """
    Implements Algorithm 11.
    """
    __init__ = partialmethod(TriangleMultiplicativeUpdate.__init__, _outgoing=True)


class TriangleMultiplicationIncoming(TriangleMultiplicativeUpdate):
    """
    Implements Algorithm 12.
    """
    __init__ = partialmethod(TriangleMultiplicativeUpdate.__init__, _outgoing=False)
