######################################################################################
#为了加速训练注释了pair embedding，并修改了 pair embedding的更新方式。
######################################################################################

from torch import nn
import torch
from typing import Tuple
from torch.nn import functional as F

import math
from torch.autograd import Variable
from write_preds_pdb import structure_build
from write_preds_pdb.geometry import Rigid, rigid_mul_vec, invert_rot_mul_vec

from model.utils1 import matrix_to_quaternion,rot_to_quat

from model.pair_embedding import PairEmbedder
from model.utils1 import permute_final_dims, flatten_final_dims, ipa_point_weights_init_, rbf, quaternions
from primitives import LayerNorm

def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    assert len(timesteps.shape) == 1
    assert embedding_dim % 2 ==0
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class SinusoidalPositionalEncoding(nn.Module):
    # Implement PE function
    def __init__(self, d_model, dropout, max_len =1000):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout) # dropout = 0.2 or 0.1
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1) # (1000) -> (1000,1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)) 
        pe[:, 0::2] = torch.sin(position * div_term) #
        pe[:, 1::2] = torch.cos(position * div_term) #
        pe = pe.unsqueeze(0) # (1000,512)->(1,1000,512) 
        self.register_buffer('pe', pe) # trainable parameters of the model
    
    def forward(self, x):
         x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
         return self.dropout(x)

class AngleEmbedder(nn.Module):
    """
    Embeds the "template_angle_feat" feature.

    Implements Algorithm 2, line 7.
    """

    def __init__(
        self,
        c_in: int,
        c_out: int,
    ):
        """
        Args:
            c_in:
                Final dimension of "template_angle_feat"
            c_out:
                Output channel dimension
        """
        super(AngleEmbedder, self).__init__()

        self.c_out = c_out
        self.c_in = c_in

        self.linear_1 = nn.Linear(self.c_in, self.c_out)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(self.c_out, self.c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [*, N_res, c_in] "template_angle_feat" features
        Returns:
            x: [*, N_res, C_out] embedding
        """
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)

        return x

class InputEmbedder(nn.Module):
    def __init__(
        self,
        nf_dim: int, # Node features dim
        c_n: int,  # Node_embedding dim
        relpos_k: int, # Window size of relative position

        # Pair Embedder parameter
        pair_dim: int, # Pair features dim
        c_z: int, # Pair embedding dim
        c_hidden_tri_att: int,
        c_hidden_tri_mul: int,
        no_blocks: int,
        no_heads: int,
        pair_transition_n: int,

        no_rigid,
    ):
        super(InputEmbedder, self).__init__()
        self.nf_dim = nf_dim

        self.c_z = c_z
        self.c_n = c_n

        self.no_rigid = no_rigid

        # self.angle_embedder = AngleEmbedder()
        '''#关掉pair-embedding
        
        self.pair_embedder = PairEmbedder(
                                        pair_dim,
                                        c_z,
                                        c_hidden_tri_att,
                                        c_hidden_tri_mul,
                                        no_blocks,
                                        no_heads,
                                        pair_transition_n)
        '''

        #self.linear_tf_z_i = nn.Linear(nf_dim, c_z)
        #self.linear_tf_z_j = nn.Linear(nf_dim, c_z)
        self.linear_tf_n = nn.Linear(nf_dim, c_n)

        # Relative_position encoding
        #self.relpos_k = relpos_k
        #self.no_bins = 2 * relpos_k + 1
        #self.linear_relpos = nn.Linear(self.no_bins, c_z)
        self.pos_encoding = SinusoidalPositionalEncoding(c_n, 0.1)

    def relpos(self,
        seq_len: int,
        batch_size: int):

        rigid_res_idx = torch.arange(0, seq_len).unsqueeze(-1).repeat(1,5).reshape(-1)
        d = rigid_res_idx - rigid_res_idx[..., None]
        boundaries = torch.arange(start=-self.relpos_k, end=self.relpos_k + 1, device=d.device)
        reshaped_bins = boundaries.view(((1,) * len(d.shape)) + (len(boundaries),))

        d = d[..., None] - reshaped_bins
        d = torch.abs(d)
        d = torch.argmin(d, dim=-1)
        d = nn.functional.one_hot(d, num_classes=len(boundaries)).float()
        l = len(d.shape)
        d = d.unsqueeze(0).repeat(batch_size,*(1,)*l).to('cpu') #  [B, N_rigid, N_rigid, C_pair]
        return d # [B, N_rigid, N_rigid, C_pair]

    def forward(self,
    #    noised_angles: torch.Tensor, #[batch,128,4]
        seq_esm: torch.Tensor, #[batch,128,320]
        rigid_type: torch.Tensor, #[batch,128,5,20]
        rigid_property: torch.Tensor, #[batch,128,5,6]

        #distance: torch.Tensor, # [batch, N_rigid, N_rigid] distance 也要做分块处理比较好 （做了_rbf）
        #altered_direction: torch.Tensor, # [batch, N_rigid, N_rigid, 3]
        #orientation: torch.Tensor,# [batch, N_rigid, N_rigid] Rigid 要把这个东西变成 quaternion
        rigid_mask: torch.Tensor, # [batch, N_rigid]  mask of the missing rigid body
        #pair_mask: torch.Tensor, # [batch, N_rigid, N_rigid]

        timesteps: torch.Tensor, # [batch, 1]
        ):
        
        assert rigid_property.shape[-1] == 6

        batch_size, seq_len, _ = seq_esm.shape

        # [batch, N_rigid, c]
        flat_rigid_type = rigid_type.reshape(batch_size, -1, rigid_type.shape[-1])
        flat_rigid_property = rigid_property.reshape(batch_size, -1, rigid_property.shape[-1])
        expand_seq = seq_esm.repeat(1,1,5).reshape(batch_size, -1, seq_esm.shape[-1])

        n_rigid = expand_seq.shape[1]

        # [batch, N_rigid, 8]
       # sin_cos = torch.cat((torch.sin(noised_angles), torch.cos(noised_angles)), -1)
       # expand_angle = sin_cos.repeat(1,1,5).reshape(batch_size, -1, sin_cos.shape[-1])

        # 直接把 time emb 成 c_n, c_z 两个维度 分别 + 到 node_emb 还有 pair_emb
        node_time = torch.tile(get_timestep_embedding(timesteps.squeeze(dim=-1), self.c_n)[:, None, :], (1, n_rigid, 1))
        '''#关掉pair-embedding
        pair_time = torch.tile(get_timestep_embedding(timesteps.squeeze(dim=-1), self.c_z)[:, None, None, :], (1, n_rigid, n_rigid, 1))

        '''

        """ 目前我们直接把 Angle 拼上去 然后做一个linear，后期可以尝试把 angle 单独拿出来 
        做一个linear，relu，linear 然后再拼回去试一下"""
        # [batch, N_rigid, nf_dim] 8 + 6 + 20 + 320,
        node_feature = torch.cat((expand_seq,flat_rigid_type, flat_rigid_property), dim=-1)

        # [*, N_rigid, c_n]
        node_feature = node_feature.float()
        node_emb = self.linear_tf_n(node_feature)
        
        # add time encode
        node_emb =  self.pos_encoding(node_emb)
        
        node_emb = node_emb + node_time #增加了位置编码，sin cos
        node_emb = node_emb * (rigid_mask[..., None].to('cpu'))

        ################ Pair_feature ####################

        # [batch, N_rigid, N_rigid, C_x] C_x = 23?
        #distance_rbf = rbf(distance)
        #orientation_quaternions = matrix_to_quaternion(orientation)
        #pair_feature = torch.cat((distance_rbf, altered_direction, orientation_quaternions),dim=-1)
        #pair_feature = pair_feature.float()
        
        #nf_emb_i = self.linear_tf_z_i(node_feature)
        #nf_emb_j = self.linear_tf_z_j(node_feature)

        # [*, N_rigid, N_rigid, c_z]
        #d = self.relpos(seq_len, batch_size)
        #relative_pos = self.linear_relpos(d)
        # [*, N_rigid, N_rigid, c_z] = [*, N_rigid, N_rigid, c_z]+ [*, N_rigid, 1, c_z] + [*, 1, N_rigid, c_z]
        #nf_pair_emb = nf_emb_i[..., None, :] + nf_emb_j[..., None, :, :]
      
        #pair_emb = self.pair_embedder(pair_feature, pair_time, relative_pos)
        
        '''
        if torch.isnan(pair_feature).any():
           print(" pair_feature张量包含NaN")
        if torch.isnan(node_feature).any():
           print("node_feature张量包含NaN")
        if torch.isnan(distance).any():
           print("distance张量包含NaN")
        if torch.isnan(distance_rbf).any():
           print("distance_rbf张量包含NaN")
        if torch.isnan(pair_emb).any():
           print(" pair_time张量包含NaN")
        if torch.isnan(pair_time).any():
           print("pair_time张量包含NaN")
        if torch.isnan(d).any():
           print("d张量包含NaN")
           print(d)
           raise ValueError("d数据中包含 NaN，请中断程序执行")  
        if torch.isnan(relative_pos).any():
           print("relative_pos张量包含NaN")
          # print(relative_pos)
           raise ValueError("relative_pos数据中包含 NaN，请中断程序执行") 
        '''
        
        
        return node_emb

class InvariantPointAttention(nn.Module):

    def __init__(
            self,
            c_n: int, # node dim
            c_z: int, # edge dim
            c_hidden: int, # ipa dim = 12
            no_heads: int, # 8
            no_qk_points: int, # 4
            no_v_points: int, # 8
            inf: float = 1e5,
            eps: float = 1e-7,
    ):
        super(InvariantPointAttention, self).__init__()

        self.c_n = c_n # node dim
        self.c_z = c_z # edge dim
        self.c_hidden = c_hidden # ipa dim = 16
        self.no_heads = no_heads # 8
        self.no_qk_points = no_qk_points # 4
        self.no_v_points = no_v_points # 8
        self.inf = inf
        self.eps = eps

        # These linear layers differ from  Alphafold IPA module, Here we use standard nn.Linear initialization
        hc = self.c_hidden * self.no_heads
        self.linear_q = nn.Linear(self.c_n, hc)
        self.linear_kv = nn.Linear(self.c_n, 2 * hc)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = nn.Linear(self.c_n, hpq)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = nn.Linear(self.c_n, hpkv)


        #self.linear_b = nn.Linear(self.c_z, self.no_heads)

        self.head_weights = nn.Parameter(torch.zeros(no_heads))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim = self.no_heads * (
            self.c_hidden + self.no_v_points * 4 #self.c_z + self.c_hidden + self.no_v_points * 4
        )
        self.linear_out = nn.Linear(concat_out_dim, self.c_n)

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    def forward(
            self,
            s: torch.Tensor, # node_emb
         #   z: torch.Tensor, # pair_emb
            r: Rigid, # rigids
            pair_mask: torch.Tensor # pair_mask
    ) -> torch.Tensor:
        """
        Args:
            s:
                [*, N_rigid, c_n] single representation
            z:
                [*, N_rigid, N_rigid, C_z] pair representation
            r:
                [*, N_rigid] transformation object
            pair_mask:
                [*, N_rigid, N_rigid] mask
        Returns:
            [*, N_res, c_n] single representation update
        """

        #z = [z]

        #######################################
        # Generate scalar and point activations
        #######################################
        # [*, N_rigid, H * C_hidden]
        q = self.linear_q(s)
        kv = self.linear_kv(s)

        # [*, N_rigid, no_heads, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_rigid, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

        # [*, N_rigid, H, C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        # [*, N_rigid, H * P_q * 3]
        q_pts = self.linear_q_points(s)

        # This is kind of clunky, but it's how the original does it
        # [*, N_rigid, H * P_q, 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1) # [*, N_rigid, H * P_q, 3]
        # q_pts = r[..., None].apply(q_pts)
        q_pts = rigid_mul_vec(r[..., None], q_pts) # rigid mut vec [*, N_rigid, 1] rigid * [*, N_rigid, H * P_q, 3]

        # [*, N_rigid, H, P_q, 3]
        q_pts = q_pts.view(
            q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3)
        )

        # [*, N_rigid, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(s)

        # [*, N_rigid, H * (P_q + P_v), 3]
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        # kv_pts = r[..., None].apply(kv_pts)
        kv_pts = rigid_mul_vec(r[..., None], kv_pts)

        # [*, N_rigid, H, (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))

        # [*, N_res, H, P_q/P_v, 3]
        k_pts, v_pts = torch.split(
            kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
        )

        ##########################
        # Compute attention scores
        ##########################
        # [*, N_rigid, N_rigid, H]
        #b = self.linear_b(z[0])

        # [*, H, N_rigid, N_rigid]
        qT_k = torch.matmul(
            permute_final_dims(q, [1, 0, 2]),  # [*, H, N_rigid, C_hidden]
            permute_final_dims(k, [1, 2, 0]),  # [*, H, C_hidden, N_rigid]
        )

        qT_k *= math.sqrt(1.0 / (3 * self.c_hidden)) # (3 * self.c_hidden) WL * c
        #b = (math.sqrt(1.0 / 3) * permute_final_dims(b, [2, 0, 1])) # [*,H, N_rigid, N_rigid]

        a = qT_k

        # [*, N_rigid, N_rigid, H, P_q, 3] = [*, N_rigid, 1, *] - [*, 1, N_rigid, *]
        pt_att = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)

        pt_att = pt_att ** 2

        # [*, N_rigid, N_rigid, H, P_q]
        pt_att = sum(torch.unbind(pt_att, dim=-1)) # calculate vector length
        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        )

        pt_att = pt_att * head_weights

        # [*, N_rigid, N_rigid, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5) # Sum over point
        # [*, N_rigid, N_rigid]
        square_mask = self.inf * (pair_mask - 1) # 这里靠 mask 逼近 -inf 之后再用 softmax 让 attention score 变 0

        # [*, H, N_rigid, N_rigid]
        pt_att = permute_final_dims(pt_att, [2, 0, 1])

        # [*, H, N_rigid, N_rigid]
        a = a + pt_att
        a = a.to('cpu') + square_mask.unsqueeze(-3).to('cpu')
        a = self.softmax(a)

        ################
        # Compute output
        ################
        # [*, N_rigid, H, C_hidden] = [*, H, N_rigid, N_rigid] matmul [*,  H, N_rigid, C_hidden]
        o = torch.matmul(
            a, v.transpose(-2, -3).to(dtype=a.dtype)
        ).transpose(-2, -3)

        # [*, N_rigid, H * C_hidden]
        o = flatten_final_dims(o, 2)


        # [*, H, 3, N_rigid, P_v]
        o_pt = torch.sum(
            (
                    a[..., None, :, :, None] # [*, H, 1, N_rigid, N_rigid, 1]
                    * permute_final_dims(v_pts, [1, 3, 0, 2])[..., None, :, :] # [*,  H, 3, 1, N_rigid, P_v]
            ),
            dim=-2, # sum over j, the second N_rigid
        )

        # [*, N_rigid, H, P_v, 3]
        o_pt = permute_final_dims(o_pt, [2, 0, 3, 1])
        # o_pt = r[..., None, None].invert_apply(o_pt)
        o_pt = invert_rot_mul_vec(r[..., None, None], o_pt)

        # [*, N_rigid, H * P_v]
        o_pt_norm = flatten_final_dims(
            torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps), 2
        )

        # [*, N_rigid, H * P_v, 3]
        o_pt = o_pt.reshape((*o_pt.shape[:-3], -1, 3))


        # [*, N_rigid, H, C_z]
        #o_pair = torch.matmul(a.transpose(-2, -3), z[0].to(dtype=a.dtype))

        # [*, N_rigid, H * C_z]
        #o_pair = flatten_final_dims(o_pair, 2)

        # [*, N_rigid, c_n]  [*, N_rigid, H * C_hidden + H * P_v * 3 + H * P_v + H * C_z]
        s = self.linear_out(
            torch.cat(
                (o, *torch.unbind(o_pt, dim=-1), o_pt_norm), dim=-1
            ).float()
        )

        return s

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

        s = self.layer_norm(s)

        return s
    
class EdgeUpdate(nn.Module):
    def __init__(self,
                 pair_dim,
                 c_z,
                 ):
        super(EdgeUpdate, self).__init__()

        self.linear_1 = nn.Linear(pair_dim, c_z)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(c_z, c_z)
        self.ln = nn.LayerNorm(c_z)

    def forward(self,
                distance: torch.Tensor,
                altered_direction: torch.Tensor,
                orientation: torch.Tensor,
                pair_time: torch.Tensor,
                relative_pos: torch.Tensor
                )-> torch.Tensor:

        distance_rbf = rbf(distance)
        orientation_quaternions = matrix_to_quaternion(orientation)
       # print("=====================================orientation=================================",orientation)
       # print("=====================================orientation_quaternions=================================",orientation_quaternions)

        #orientation_quaternions = rot_to_quat(orientation)
        #orientation_quaternions1 = quaternions(orientation)
        pair_feature = torch.cat((distance_rbf, altered_direction, orientation_quaternions), dim=-1)
        pair_emb = pair_feature.float()
        pair_emb = self.linear_1(pair_emb)
        pair_emb = self.relu(pair_emb)
        pair_emb = self.linear_2(pair_emb)

        pair_emb = pair_emb + pair_time + relative_pos

        pair_emb = self.ln(pair_emb)

        return pair_emb

'''
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

    def forward(self, node_emb, edge_emb):
        # [batch, N_rigid, c_n/2]
        node_emb = self.initial_embed(node_emb)
        batch_size, num_rigids, _ = node_emb.shape
        # [batch, N_rigid, N_rigid, c_n]
        edge_bias = torch.cat([
            torch.tile(node_emb[:, :, None, :], (1, 1, num_rigids, 1)),
            torch.tile(node_emb[:, None, :, :], (1, num_rigids, 1, 1)),
        ], axis=-1)
        # [batch * N_rigid * N_rigid, c_n + c_z]
        edge_emb = torch.cat(
            [edge_emb, edge_bias], axis=-1).reshape(
                batch_size * num_rigids**2, -1)

        # [batch * N_rigid * N_rigid, c_z]
        edge_emb1 = self.trunk(edge_emb)
        edge_emb = self.final_layer(edge_emb1 + edge_emb)
        edge_emb = self.layer_norm(edge_emb)

        # [batch, N_rigid, N_rigid, c_z]
        edge_emb = edge_emb.reshape(
            batch_size, num_rigids, num_rigids, -1
        )
        return edge_emb
'''

class AngleNoise(nn.Module):

    def __init__(self, c_in, c_hidden, no_blocks, no_angles, no_rigids, epsilon):
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
        
        super(AngleNoise, self).__init__()

        self.c_in = c_in
        self.no_rigids = no_rigids
        self.c_hidden = c_hidden
        self.no_blocks = no_blocks
        self.no_angles = no_angles
        self.eps = epsilon

        self.linear_in = nn.Linear(self.c_in * self.no_rigids, self.c_hidden)
        self.linear_initial = nn.Linear(self.c_in * self.no_rigids, self.c_hidden)

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
                [*, N_rigid, c_n] single embedding
            s_initial:
                [*, N_rigid, c_n] single embedding as of the start of the
                StructureModule
        Returns:
            [*, no_angles] predicted angles
        """
        # NOTE: The ReLU's applied to the inputs are absent from the supplement
        # pseudocode but present in the source. For maximal compatibility with
        # the pretrained weights, I'm going with the source.

        # [*, N_res, c_n * no_rigid]
        # 这里把不同 rigid 的信息拼起来在这里
        s_initial = s_initial.reshape(s_initial.shape[0], -1, s_initial.shape[-1] * self.no_rigids)
        s = s.reshape(s.shape[0], -1, s.shape[-1] * self.no_rigids)

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

class AngleResnet(nn.Module):
    """
    Implements Algorithm 20, lines 11-14
    """

    def __init__(self, c_in, c_hidden, no_blocks, no_angles, no_rigids, epsilon):
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
        super(AngleResnet, self).__init__()

        self.c_in = c_in 
        self.no_rigids = no_rigids
        self.c_hidden = c_hidden
        self.no_blocks = no_blocks
        self.no_angles = no_angles
        self.eps = epsilon

        self.linear_in = nn.Linear(self.c_in * self.no_rigids, self.c_hidden)
        self.linear_initial = nn.Linear(self.c_in * self.no_rigids, self.c_hidden)

        self.layers = nn.ModuleList()
        for _ in range(self.no_blocks):
            layer = AngleResnetBlock(c_hidden=self.c_hidden)
            self.layers.append(layer)

        self.linear_out = nn.Linear(self.c_hidden, self.no_angles * 2)

        self.relu = nn.ReLU()

    def forward(
        self, s: torch.Tensor, s_initial:torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s:
                [*, N_rigid, c_n] single embedding
            s_initial:
                [*, N_rigid, c_n] single embedding as of the start of the
                StructureModule
        Returns:
            [*, no_angles, 2] predicted angles
        """
        # NOTE: The ReLU's applied to the inputs are absent from the supplement
        # pseudocode but present in the source. For maximal compatibility with
        # the pretrained weights, I'm going with the source.



        # [*, N_res, c_n * no_rigid]
        # 这里把不同 rigid 的信息拼起来在这里
        s_initial = s_initial.reshape(s_initial.shape[0], -1, s_initial.shape[-1] * self.no_rigids)
        s = s.reshape(s.shape[0], -1, s.shape[-1] * self.no_rigids)

        # [*, N_res, C_hidden]
        s_initial = self.relu(s_initial)
        s_initial = self.linear_initial(s_initial)
        s = self.relu(s)
        s = self.linear_in(s)
        s = s + s_initial

        for l in self.layers:
            s = l(s)

        s = self.relu(s)

        # [*, N_res, no_angles * 2]
        s = self.linear_out(s)
      #  print("============s==================",s.shape)

        # [*, N_res, no_angles, 2]
        s = s.view(s.shape[:-1] + (-1, 2))
      #  print("============s==================",s.shape)
        norm_denom = torch.sqrt(
            torch.clamp(torch.sum(s ** 2, dim=-1, keepdim=True),min=self.eps)
        )      
        s = s / norm_denom     
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

class StructureUpdateModule(nn.Module):

    def __init__(self,
                 no_blocks,
                 c_n,
                 c_z,
                 pair_dim,
                 c_hidden,
                 ipa_no_heads,
                 no_qk_points,
                 no_v_points,
                 c_resnet,
                 no_resnet_blocks,
                 no_angles,
                 no_rigids,
                 epsilon,
                 ):

        super(StructureUpdateModule, self).__init__()
        self.no_blocks = no_blocks
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
            
        self.angle_resnet = AngleResnet(c_n,
                                        c_resnet,
                                        no_resnet_blocks,
                                        no_angles,
                                        no_rigids,
                                        epsilon)
        '''
        self.edge_transition = EdgeTransition(c_n,
                                              c_z,
                                              c_z)
        '''
        #self.edge_update = EdgeUpdate(pair_dim,
        #                             c_z)
        
    def forward(self,
                seq_idx,
                backbone_coords,
             #   pair_time,
             #   relative_pos,
                 init_node_emb,
             #   pair_emb,
                rigids,
                pair_mask,
            #    pad_mask,
                ):

        node_emb = torch.clone(init_node_emb)
        for i, block in enumerate(self.blocks):
            node_emb = block(node_emb,rigids,pair_mask)
            
            #if torch.isnan(node_emb).any():
            #    print("node_emb 张量包含NaN",i)
            

            
            
            updated_chi_angles = self.angle_resnet(node_emb,init_node_emb)
            
            rigids = structure_build.torsion_to_frame(seq_idx, backbone_coords, updated_chi_angles)
           # _, _, distance, altered_direction, orientation = structure_build.frame_to_edge(rigids, seq_idx, pad_mask)
            #pair_emb = self.edge_update(distance, altered_direction, orientation, pair_time, relative_pos)
            
            #pair_emb1 = pair_emb1*pair_mask.unsqueeze(-1)
            #rigids = rigids*flat_mask
            #print('=======================',rigids.shape)
            #print('=======================',pair_emb1)
            #print("============================================",i)
           # print("=================distance========================", distance)
           # print("=================altered_direction========================",altered_direction)
           #$ print("=================orientation========================", orientation)
           # print("=================quaternions(orientation)========================",quaternions(orientation).float())
           
            #print("============rbf===========",rbf(distance))
            '''
            if torch.isnan(pair_time).any():
                print(" pair_time张量包含NaN")
               
            if torch.isnan(relative_pos).any():
                print("relative_pos张量包含NaN")
                raise ValueError("altered_direction)数据中包含 NaN，请中断程序执行")
            if torch.isnan(orientation).any():
                print("orientation张量包含NaN")
                raise ValueError("orientation数据中包含 NaN，请中断程序执行")
            if torch.isnan(updated_chi_angles).any():
                print("updated_chi_angles张量包含NaN")
                raise ValueError("quaternions(orientation)数据中包含 NaN，请中断程序执行")
            '''
            
      
        return updated_chi_angles

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

        # [*, N_res, c_n * 5]
        self.ipa = InvariantPointAttention(c_n,
                                           c_z,
                                           c_hidden,
                                           ipa_no_heads,
                                           no_qk_points,
                                           no_v_points)

        self.ipa_ln = LayerNorm(c_n)

        '''
        self.skip_embed = nn.Linear(
            self._model_conf.node_embed_size,
            self._ipa_conf.c_skip,
            init="final"
        )
        '''

        self.node_transition = TransitionLayer(c_n)


    def forward(self,
                node_emb,
                rigids,
                pair_mask
                ):

        # [*, N_rigid, c_n]
        ipa_emb = self.ipa(node_emb, rigids,pair_mask)
        node_emb = self.ipa_ln(node_emb + ipa_emb)
        node_emb = self.node_transition(node_emb)
        return node_emb
class RigidDiffusion(nn.Module):

    def __init__(self,
                 num_blocks: int = 3, # StructureUpdateModule的循环次数

                 # InputEmbedder config
                 nf_dim: int = 6 + 20 + 320, # 8 +6 + 20 + 320
                 c_n: int = 384, # Node channel dimension after InputEmbedding
                 relpos_k: int = 32, # relative position neighbour range

                 # PairEmbedder parameter
                 pair_dim: int = 23, # rbf + direction_vector + qu
                 c_z: int = 64, # Pair channel dimension after InputEmbedding
                 c_hidden_tri_att: int = 16, # x2 cause we x2 the input dimension
                 c_hidden_tri_mul: int = 32, # Keep ori
                 pairemb_no_blocks: int = 2, # Keep ori
                 mha_no_heads: int = 4, # Keep ori
                 pair_transition_n: int = 2, # Keep ori

                 # IPA config
                 c_hidden: int = 12,  # IPA hidden channel dimension
                 ipa_no_heads: int = 8,  # Number of attention head
                 no_qk_points: int =4,  # Number of qurry/key (3D vector) point head
                 no_v_points: int =8,  # Number of value (3D vector) point head

                 # AngleResnet
                 c_resnet: int = 128, # AngleResnet hidden channel dimension
                 no_resnet_blocks: int = 2, # Resnet block number
                 no_angles: int = 4, # predict chi 1-4 4 angles
                 no_rigids: int = 5, # number of rigids to concate togather
                 epsilon: int = 1e-7,

                 no_rigid: int = 5,
                 ):

        super(RigidDiffusion, self).__init__()

        self.num_blocks = num_blocks
        
        self.input_embedder = InputEmbedder(nf_dim, c_n, relpos_k, # Node feature related dim
                                            pair_dim, c_z, # Pair feature related dim
                                            c_hidden_tri_att, c_hidden_tri_mul, # hidden dim for TriangleAttention, TriangleMultiplication
                                            pairemb_no_blocks, mha_no_heads, pair_transition_n,
                                            no_rigid)

        self.structure_update = StructureUpdateModule(num_blocks,
                                                     c_n,
                                                     c_z,
                                                     pair_dim,
                                                     c_hidden,
                                                     ipa_no_heads,
                                                     no_qk_points,
                                                     no_v_points,
                                                     c_resnet,
                                                     no_resnet_blocks,
                                                     no_angles,
                                                     no_rigids,
                                                     epsilon
        )

        '''
        self.noise_predictor_sincos = AngleResnet(c_n,
                                                  c_resnet,
                                                  no_resnet_blocks,
                                                  no_angles,
                                                  no_rigids,
                                                  epsilon
        )
        
        # 预测8个 sin cos  改为预测4个角度
        self.noise_predictor = AngleNoise(c_n,
                                           c_resnet,
                                           no_resnet_blocks,
                                           no_angles, 
                                           no_rigids,
                                           epsilon
        )
        '''

    def forward(self,
                side_chain_angles,
                backbone_coords,
                seq_idx,
                timesteps,
                seq_esm,
                rigid_type,
                rigid_property,
                pad_mask,
        ):
  
        
        

        # [*, N_rigid, 4, 2]
        angles_sin_cos = torch.stack([torch.sin(side_chain_angles), torch.cos(side_chain_angles)], dim=-1)
        # [*, N_rigid] Rigid
        rigids = structure_build.torsion_to_frame(seq_idx,
                                                 backbone_coords,
                                                 angles_sin_cos)  # add attention #frame
        # flat_mask [*, N_rigid]， others [*, N_rigid, N_rigid, c]
        pair_mask, rigid_mask, distance, altered_direction, orientation = structure_build.frame_to_edge(rigids, seq_idx, pad_mask)
                
        # [*, N_rigid, c_n], [*, N_rigid, N_rigid, c_z]
        init_node_emb = self.input_embedder(
                                                  seq_esm,
                                                  rigid_type,
                                                  rigid_property,
                                                #  distance,
                                                #  altered_direction,
                                                 # orientation,
                                                  rigid_mask,
                                               #   pair_mask,
                                                  timesteps)
          
        # [*, N_res, c_n * 5]      
        pred_chi_sin_cos = self.structure_update(seq_idx,
                                        backbone_coords,
                                        #pair_time,
                                        #relative_pos,
                                        init_node_emb,
                                        #pair_emb,
                                        rigids,
                                        pair_mask,)
                                    #    pad_mask)
        #if torch.isnan(pred_chi_sin_cos).any():
        #   print("pred_chi_sin_cos张量包含NaN")
        # Reshape N_rigid into N_res 这里其实一直没有好好写， 这五个rigid的表示直接被拼起来就用来预测角度了，这里是不是应该换一种方法？
        # 直接放到 IPA 里面怎么样
        # node_emb = node_emb.reshape(node_emb.shape[0], -1, node_emb.shape[-1] * 5)
        # 或许正确的操作应该是 预测noise 用加噪音后的角度-noise 继续， 而不是直接预测角度 然后最后预测noise

        # noise = self.noise_predictor_sincos(node_emb, init_node_emb)

        #  noise = self.noise_predictor(node_emb, init_node_emb)
        #print("============pred_chi_sin_cos=================", pred_chi_sin_cos)
        return pred_chi_sin_cos

