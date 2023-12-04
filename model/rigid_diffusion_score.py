from torch import nn
import torch
from typing import Tuple
from torch.nn import functional as F
import numpy as np

import math
import torch_scatter

from model.pair_embedding_score import PairEmbedder
from primitives import LayerNorm
from model.geometric_attention import GraphIPA

def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    assert len(timesteps.shape) == 1
    assert embedding_dim % 2 ==0
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class InputEmbedder(nn.Module):
    def __init__(
        self,
        nf_dim: int, # Node features dim
        c_n: int,  # Node_embedding dim

        # Pair Embedder parameter
        pair_dim: int, # Pair features dim
        c_z: int, # Pair embedding dim
        #c_hidden_tri_att: int,
        #c_hidden_tri_mul: int,
        #no_blocks: int,
        #no_heads: int,
        #pair_transition_n: int,

        sigma_embed_dim: int,
        sigma_min = 0.01 * np.pi,
        sigma_max = np.pi,
    ):
        super(InputEmbedder, self).__init__()
        self.nf_dim = nf_dim
        self.c_z = c_z
        self.c_n = c_n
        self.sigma_embed_dim = sigma_embed_dim

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        self.pair_embedder = PairEmbedder(pair_dim,
                                        c_z,
                                        #c_hidden_tri_att,
                                        #c_hidden_tri_mul,
                                        #no_blocks,
                                        #no_heads,
                                        #pair_transition_n
                                        )


        self.linear_tf_n = nn.Linear(nf_dim, c_n - 32)

    def forward(self, data):

        node_sigma = torch.log(data.node_sigma / self.sigma_min) / np.log(self.sigma_max / self.sigma_min) * 10000
        node_time_emb = get_timestep_embedding(node_sigma, self.sigma_embed_dim)

        pair_time = node_time_emb[data.edge_index[0]]

        # [N_rigid, c_n]
        node_emb = self.linear_tf_n(data.x)
        
        # add time encode
        node_emb = torch.cat((node_emb, node_time_emb),-1)

        ################ Pair_feature ####################
        pair_emb = torch.cat((pair_time, data.edge_attr), dim=-1)
        pair_emb = self.pair_embedder(pair_emb)

        return node_emb, pair_emb

class TransitionLayer(nn.Module):
    """ We only get one transitionlayer in our model, so no module needed."""
    def __init__(self, c):
        super(TransitionLayer, self).__init__()

        self.c = c

        self.linear_1 = nn.Linear(self.c, self.c)
        self.linear_2 = nn.Linear(self.c, self.c)
        self.linear_3 = nn.Linear(self.c, self.c)

        self.relu = nn.ReLU()

        self.layer_norm = LayerNorm(self.c)

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)

        s = s + s_initial

        s = s
        s = self.layer_norm(s)

        return s

class EdgeTransition(nn.Module):
    def __init__(
            self,
            node_embed_size,
            edge_embed_in,
            edge_embed_out,
            num_layers=2,
            node_dilation=2
        ):
        super(EdgeTransition, self).__init__()

        bias_embed_size = node_embed_size // node_dilation
        self.initial_embed = nn.Linear(node_embed_size, bias_embed_size)
        hidden_size = bias_embed_size * 2 + edge_embed_in
        trunk_layers = []
        for _ in range(num_layers):
            trunk_layers.append(nn.Linear(hidden_size, hidden_size))
            trunk_layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*trunk_layers)
        self.final_layer = nn.Linear(hidden_size, edge_embed_out)
        self.layer_norm = nn.LayerNorm(edge_embed_out)

    def forward(self, data, node_emb, edge_emb):

        # [N_rigid, c_n/2]
        node_emb = self.initial_embed(node_emb)

        # [E, c_n]
        edge_bias = torch.cat([node_emb[data.edge_index[0]], node_emb[data.edge_index[1]],], axis=-1)

        # [E, c_n + c_z]
        edge_emb = torch.cat([edge_emb, edge_bias], axis=-1)

        # [E, c_z]
        edge_emb1 = self.trunk(edge_emb)
        edge_emb = self.final_layer(edge_emb1 + edge_emb)
        edge_emb = self.layer_norm(edge_emb)

        return edge_emb

class AngleScore(nn.Module):

    def __init__(self, c_n, c_hidden, no_blocks, no_angles, epsilon):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Hidden channel dimension
            no_blocks:
                Number of resnet blocks
            no_angles:
                Number of torsion angles to generate
            epsilon:
                Small constant for normalization
        """
        
        super(AngleScore, self).__init__()

        self.c_n = c_n
        self.c_hidden = c_hidden
        self.no_blocks = no_blocks
        self.no_angles = no_angles
        self.eps = epsilon

        self.linear_in = nn.Linear(self.c_n * 5, self.c_hidden)
        self.linear_initial = nn.Linear(self.c_n * 5, self.c_hidden)

        self.layers = nn.ModuleList()
        for _ in range(self.no_blocks):
            layer = AngleResnetBlock(c_hidden=self.c_hidden)
            self.layers.append(layer)

        self.linear_out = nn.Linear(self.c_hidden, self.no_angles)

        self.relu = nn.ReLU()

    def forward(
            self, s: torch.Tensor, s_initial: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        """
        Args:
            s:
                [N_res, c_n] single embedding
            s_initial:
                [N_res, c_n] single embedding as of the start of the
                StructureModule
        Returns:
            [*, no_angles] predicted angles
        """

        # NOTE: The ReLU's applied to the inputs are absent from the supplement
        # pseudocode but present in the source. For maximal compatibility with
        # the pretrained weights, I'm going with the source.

        # [*, N_res, C_hidden]
        s_initial = self.relu(s_initial)
        s_initial = self.linear_initial(s_initial)
        s = self.relu(s)
        s = self.linear_in(s)
        s = s + s_initial

        for l in self.layers:
            s = l(s)

        s = self.relu(s)

        # [*, N_res, no_angles]
        s = self.linear_out(s)

        return s

class AngleResnetBlock(nn.Module):
    def __init__(self, c_hidden):
        """
        Args:
            c_hidden:
                Hidden channel dimension
        """
        super(AngleResnetBlock, self).__init__()

        self.c_hidden = c_hidden

        self.linear_1 = nn.Linear(self.c_hidden, self.c_hidden)
        self.linear_2 = nn.Linear(self.c_hidden, self.c_hidden)

        self.relu = nn.ReLU()

    def forward(self, a: torch.Tensor) -> torch.Tensor:

        s_initial = a

        a = self.relu(a)
        a = self.linear_1(a)
        a = self.relu(a)
        a = self.linear_2(a)
        t = a + s_initial

        return t

class RigidUpdate(nn.Module):
    def __init__(self,
                 c_n):

        super(RigidUpdate, self).__init__()
        self.c_n = c_n
        self.linear_initial = nn.Linear(self.c_n, self.c_n)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(self.c_n, 3)

    def forward(self,
                s,
                ):
        """
                Args:
                    s:
                        [B, N_rigid, c_n] single embedding

                        StructureModule
                Returns:
                    [*, no_angles] predicted angles
                """


        s = self.linear_initial(s)
        s = self.relu(s)
        trans = self.linear(s)

        return trans

class StructureUpdateModule(nn.Module):

    def __init__(self,
                 no_blocks,
                 c_n,
                 c_z,
                 c_hidden,
                 ipa_no_heads,
                 no_qk_points,
                 no_v_points,
                 ):

        super(StructureUpdateModule, self).__init__()

        self.blocks = nn.ModuleList()
        self.relu = nn.ReLU()
        for _ in range(no_blocks):
            block = StructureBlock(c_n,
                                   c_z,
                                   c_hidden,
                                   ipa_no_heads,
                                   no_qk_points,
                                   no_v_points,
                                   )

            self.blocks.append(block)

        self.edge_transition = EdgeTransition(c_n,
                                              c_z,
                                              c_z)

    def forward(self, data, init_node_emb, pair_emb):

        for i, block in enumerate(self.blocks):
            node_emb = block(data, init_node_emb, pair_emb)

            pair_emb = self.edge_transition(data, node_emb, pair_emb)

        return node_emb

class StructureBlock(nn.Module):
    def __init__(self,
                 c_n,
                 c_z,
                 c_hidden,
                 ipa_no_heads,
                 no_qk_points,
                 no_v_points,
                 ):
        super(StructureBlock, self).__init__()
        self.edge_ipa = GraphIPA(c_n,
                                 c_hidden,
                                 c_z,
                                 ipa_no_heads,
                                 no_qk_points,
                                 no_v_points,
        )
        self.ipa_ln = LayerNorm(c_n)

        '''
        self.skip_embed = nn.Linear(
            self._model_conf.node_embed_size,
            self._ipa_conf.c_skip,
            init="final"
        )
        '''

        self.node_transition = TransitionLayer(c_n)


    def forward(self, data, init_node_emb, pair_emb):

        # [N, C_hidden]
        node_emb = init_node_emb + self.edge_ipa(init_node_emb, pair_emb, data)
        node_emb = self.ipa_ln(node_emb)
        node_emb = self.node_transition(node_emb)

        return node_emb

class RigidDiffusion(nn.Module):

    def __init__(self,
                 num_blocks: int = 3, # StructureUpdateModule的循环次数

                 # InputEmbedder config
                 nf_dim: int = 7 + 19 + 320,
                 c_n: int = 384, # Node channel dimension after InputEmbedding

                 # PairEmbedder parameter
                 pair_dim: int = 805, # rbf+3+4 + nf_dim* 2 + 2* relpos_k+1 + 10 edge type
                 c_z: int = 64, # Pair channel dimension after InputEmbedding
                 #c_hidden_tri_att: int = 16, # x2 cause we x2 the input dimension
                 #c_hidden_tri_mul: int = 32, # Keep ori
                 #pairemb_no_blocks: int = 2, # Keep ori
                 #mha_no_heads: int = 4, # Keep ori
                 #pair_transition_n: int = 2, # Keep ori

                 # IPA config
                 c_hidden: int = 12,  # IPA hidden channel dimension
                 ipa_no_heads: int = 8,  # Number of attention head
                 no_qk_points: int =4,  # Number of qurry/key (3D vector) point head
                 no_v_points: int =8,  # Number of value (3D vector) point head

                 # AngleResnet
                 c_resnet: int = 128, # AngleResnet hidden channel dimension
                 no_resnet_blocks: int = 2, # Resnet block number
                 no_angles: int = 4, # predict chi 1-4 4 angles
                 epsilon: int = 1e-7,
                 top_k: int =64,

                 sigma_embed_dim: int = 32,
                 # Arc config
                 all_loc = False,
                 ):

        super(RigidDiffusion, self).__init__()

        self.all_loc = all_loc
        self.num_blocks = num_blocks
        self.top_k = top_k

        self.input_embedder = InputEmbedder(nf_dim, c_n,
                                            pair_dim, c_z, # Pair feature related dim
                                            #c_hidden_tri_att, c_hidden_tri_mul, # hidden dim for TriangleAttention, TriangleMultiplication
                                            #pairemb_no_blocks, mha_no_heads, pair_transition_n,
                                            sigma_embed_dim)

        self.structure_update = StructureUpdateModule(num_blocks,
                                                     c_n,
                                                     c_z,
                                                     c_hidden,
                                                     ipa_no_heads,
                                                     no_qk_points,
                                                     no_v_points)

        self.score_predictor = AngleScore(c_n,
                                           c_resnet,
                                           no_resnet_blocks,
                                           no_angles, 
                                           epsilon
        )

    def forward(self, data):

        # [N_rigid, c_n], [e, c_z]
        init_node_emb, pair_emb = self.input_embedder(data)
          
        # [N_rigid, c_n]
        node_emb = self.structure_update(data, init_node_emb, pair_emb)
     
        # [N_res, 5, c_n]
        init_residue_emb = torch.zeros(data.rigid_mask.shape[0], init_node_emb.shape[-1]).to(node_emb.device)
        init_residue_emb[data.rigid_mask] = init_node_emb
        init_residue_emb = init_residue_emb.reshape(-1, 5, node_emb.shape[-1])
        init_residue_emb = init_residue_emb.reshape(-1, 5* node_emb.shape[-1])
        # [N_res, 5, c_n]
        residue_emb = torch.zeros(data.rigid_mask.shape[0], node_emb.shape[-1]).to(node_emb.device)
        residue_emb[data.rigid_mask] = node_emb
        residue_emb = residue_emb.reshape(-1, 5, node_emb.shape[-1])
        residue_emb = residue_emb.reshape(-1, 5* node_emb.shape[-1])

        # [N_res, c_n] -->  [N_res,4]    
        score = self.score_predictor(residue_emb, init_residue_emb)

        return score
