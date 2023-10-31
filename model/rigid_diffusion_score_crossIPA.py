######################################################################################
#为了加速训练注释了pair embedding，并修改了 pair embedding的更新方式。
######################################################################################

from torch import nn
import torch
from typing import Tuple
from torch.nn import functional as F
import numpy as np

import math
from torch.autograd import Variable
from write_preds_pdb import structure_build_score, constant
from write_preds_pdb.geometry import Rigid, loc_rigid_mul_vec, loc_invert_rot_mul_vec, Rotation, Rigid_update_trans, rot_vec

from model.utils1 import matrix_to_quaternion,rot_to_quat

from model.pair_embedding_score import PairEmbedder
from model.utils1 import permute_final_dims, flatten_final_dims, ipa_point_weights_init_, rbf, quaternions
from primitives import LayerNorm


def gather_node(nodes, neighbor_idx):
    """
    [B, N, H, (P_q + P_v), 3] => [B, N, K, H, (P_q + P_v), 3]
    [*, N_rigid, K]
    # [*, N_rigid, H, 2 * C_hidden] => [*, N_rigid, K, H, 2 * C_hidden]
    """

    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]

    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1, *(1,)*(len(nodes.shape)-len(neighbor_idx.shape)+ 1)))
    neighbors_flat = neighbors_flat.expand(-1, -1, *nodes.shape[2:])
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + list(nodes.shape)[2:])
    return neighbor_features

def gather_edges(edges, neighbor_idx):
    """
    Z = [*, N_rigid, N_rigid, C_z]
    neighbor_idx [*, N_rigid, K]
    """
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features

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
        edge_type: int,
        pair_dim: int, # Pair features dim
        c_z: int, # Pair embedding dim
        c_hidden_tri_att: int,
        c_hidden_tri_mul: int,
        no_blocks: int,
        no_heads: int,
        pair_transition_n: int,

        no_rigids,
    ):
        super(InputEmbedder, self).__init__()
        self.nf_dim = nf_dim

        self.c_z = c_z
        self.c_n = c_n

        self.no_rigids = no_rigids

        # self.angle_embedder = AngleEmbedder()
        #关掉pair-embedding
        
        self.pair_embedder = PairEmbedder(
                                        pair_dim,
                                        c_z,
                                        c_hidden_tri_att,
                                        c_hidden_tri_mul,
                                        no_blocks,
                                        no_heads,
                                        pair_transition_n)


        self.linear_tf_n = nn.Linear(nf_dim, c_n)

        # Relative_position encoding
        self.relpos_k = relpos_k
        self.no_bins = 2 * relpos_k + 1
        self.edge_type = edge_type

    def relpos(self,
        seq_len: int,
        batch_size: int):

        rigid_res_idx = torch.arange(0, seq_len).unsqueeze(-1).repeat(1,5).reshape(-1)
        d = rigid_res_idx - rigid_res_idx[..., None]
        boundaries = torch.arange(start=-self.relpos_k, end=self.relpos_k + 1, device=d.device)
        reshaped_bins = boundaries.view(((1,) * len(d.shape)) + (len(boundaries),))

        mask = d ==0
        reverse_mask = mask ==0

        d = d[..., None] - reshaped_bins
        d = torch.abs(d)
        d = torch.argmin(d, dim=-1)
        d = nn.functional.one_hot(d, num_classes=len(boundaries)).float()
        l = len(d.shape)
        d = d.unsqueeze(0).repeat(batch_size,*(1,)*l).to('cuda') #  [B, N_rigid, N_rigid, C_pair]

        rigid_rigid_idx = torch.arange(0, 5).repeat(1, seq_len).reshape(-1).to('cuda') 
        rigid_edge = rigid_rigid_idx - rigid_rigid_idx[..., None]
        rigid_edge = rigid_edge * mask.to('cuda')  + 5* reverse_mask.to('cuda')  + torch.abs(torch.min(rigid_edge))
        rigid_edge = nn.functional.one_hot(rigid_edge, num_classes=self.edge_type).float()
        rigid_edge = rigid_edge.unsqueeze(0).repeat(batch_size, *(1,) * len(rigid_edge.shape))

        d = torch.cat([d,rigid_edge], dim=-1)
        return d # [B, N_rigid, N_rigid, relpos_k + 10]

    def forward(self,
        #noised_angles: torch.Tensor, #[batch,128,4]
        seq_esm: torch.Tensor, #[batch,128,320]
        # diffusion_mask: torch.Tensor, #[batch,128,1]
        rigid_type: torch.Tensor, #[batch,128,5,20]
        rigid_property: torch.Tensor, #[batch,128,5,6]

        distance: torch.Tensor, # [batch, N_rigid, N_rigid] distance 也要做分块处理比较好 （做了_rbf）
        #altered_direction: torch.Tensor, # [batch, N_rigid, N_rigid, 3]
        #orientation: torch.Tensor,# [batch, N_rigid, N_rigid] Rigid 要把这个东西变成 quaternion
        rigid_mask: torch.Tensor, # [batch, N_rigid]  mask of the missing rigid body
        pair_mask: torch.Tensor, # [batch, N_rigid, N_rigid]
        E_idx: torch.Tensor,

        sigma: torch.Tensor, # [batch, 1]
        sigma_min = 0.01 * torch.pi,
        sigma_max = torch.pi,

        ):
        
        batch_size, seq_len, _ = seq_esm.shape
        n_rigid = rigid_mask.shape[1]

        # [batch, N_rigid, c]
        flat_rigid_type = rigid_type.reshape(batch_size, -1, rigid_type.shape[-1])
        flat_rigid_property = rigid_property.reshape(batch_size, -1, rigid_property.shape[-1])
        expand_seq = seq_esm.repeat(1,1,5).reshape(batch_size, -1, seq_esm.shape[-1])


        # [batch, N_rigid, 8]
        # sin_cos = torch.cat((torch.sin(noised_angles), torch.cos(noised_angles)), -1)
        # expand_angle = sin_cos.repeat(1,1,5).reshape(batch_size, -1, sin_cos.shape[-1])

        # [batch, N_res, 5]
        #expand_diffusion_mask =  diffusion_mask.repeat(1,1,5)
        #expand_diffusion_mask[...,0] = False
        expand_diffusion_mask = torch.ones((batch_size,seq_len,5)) == 1
        expand_diffusion_mask[..., 0] = False

        # [batch, N_rigid, c_n/2]
        expand_diffusion_mask = expand_diffusion_mask.reshape(batch_size, -1, 1).repeat(1,1,self.c_n//2).to('cuda')
        mask_time = torch.cat([torch.sin(expand_diffusion_mask), torch.cos(expand_diffusion_mask)], dim=-1)
        node_sigma = torch.log(sigma / sigma_min) / np.log(sigma_max / sigma_min) * 10000
        node_time = torch.tile(get_timestep_embedding(node_sigma, self.c_n)[:, None, :], (1, n_rigid, 1))
        node_time = torch.where(expand_diffusion_mask.repeat(1,1,2), node_time, mask_time)


        '''#关掉pair-time
        pair_time = torch.tile(get_timestep_embedding(timesteps.squeeze(dim=-1), self.c_z)[:, None, None, :], (1, n_rigid, n_rigid, 1))

        '''

        # [batch, N_rigid, nf_dim] 6 + 20 + 320,
        node_feature = torch.cat((expand_seq, flat_rigid_type, flat_rigid_property), dim=-1)

        # [*, N_rigid, c_n]
        node_feature = node_feature.float()
        node_emb = self.linear_tf_n(node_feature)
        
        # add time encode
        node_emb = node_emb + node_time
        node_emb = node_emb * (rigid_mask[..., None].to('cuda'))

        ################ Pair_feature ####################

        # [batch, N_rigid, K, C_x] C_x = 23?
        distance_e = gather_edges(distance[...,None], E_idx).squeeze(-1)
        distance_rbf = rbf(distance_e)

        #quaternions = matrix_to_quaternion(orientation)
        #quaternions = gather_edges(quaternions, E_idx)

        #altered_direction = gather_edges(altered_direction, E_idx)
        
        nf_pair_feature = torch.cat([torch.tile(node_feature[:, :, None, :], (1, 1, n_rigid, 1)),
                                     torch.tile(node_feature[:, None, :, :], (1, n_rigid, 1, 1))], axis = -1)
        nf_pair_feature = gather_edges(nf_pair_feature, E_idx)

        # [*, N_rigid, K, c_z]
        d = self.relpos(seq_len, batch_size)
        relative_pos = gather_edges(d, E_idx)

        # [*, N_rigid, K]
        pair_mask_e = gather_edges(pair_mask.unsqueeze(-1), E_idx).squeeze(-1)

        pair_feature = torch.cat([distance_rbf, 
                                  #altered_direction,
                                  #quaternions,
                                  nf_pair_feature, 
                                  relative_pos],
                                  dim = -1).float()
        
        pair_emb = self.pair_embedder(pair_feature, pair_mask_e)

        return node_emb, pair_emb, pair_mask_e

class EdgeInvariantPointAttention(nn.Module):

    def __init__(
            self,
            c_n: int,  # node dim
            c_z: int,  # edge dim
            c_hidden: int,  # ipa dim = 12
            no_heads: int,  # 8
            no_qk_points: int,  # 4
            no_v_points: int,  # 8
            inf: float = 1e8,
            eps: float = 1e-6,
    ):
        super(EdgeInvariantPointAttention, self).__init__()

        self.c_n = c_n  # node dim
        self.c_z = c_z  # edge dim
        self.c_hidden = c_hidden  # ipa dim = 16
        self.no_heads = no_heads  # 8
        self.no_qk_points = no_qk_points  # 4
        self.no_v_points = no_v_points  # 8
        self.inf = inf
        self.eps = eps

        # These linear layers differ from  Alphafold IPA module, Here we use standard nn.Linear initialization
        hc = self.c_hidden * self.no_heads
        self.linear_q = nn.Linear(self.c_n, hc, bias=False)
        self.linear_kv = nn.Linear(self.c_n, 2 * hc, bias=False)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = nn.Linear(self.c_n, hpq, bias=False)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = nn.Linear(self.c_n, hpkv, bias=False)

        self.linear_b = nn.Linear(self.c_z, self.no_heads, bias=False)

        self.head_weights = nn.Parameter(torch.zeros(no_heads))
        ipa_point_weights_init_(self.head_weights)

        concat_out_dim = self.no_heads * (
                self.c_z + self.c_hidden + self.no_v_points * 4
        )
        self.linear_out = nn.Linear(concat_out_dim, self.c_n)

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    def forward(self,
             s: torch.Tensor, # node_emb
             z_e: torch.Tensor, # [*, N_rigid, K, C_z]
             r: Rigid, # I will need to make the rigid also become neighbor???
             direction: torch.Tensor,
             rel_ori: torch.Tensor,
             pair_mask: torch.Tensor,  # pair_mask
             E_idx: torch.Tensor,
             )-> torch.Tensor:




        # [*, N_rigid, H * C_hidden]
        q = self.linear_q(s)
        # [*, N_rigid, H * C_hidden]
        kv = self.linear_kv(s)

        # [*, N_rigid, no_heads, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_rigid, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

        # [*, N_rigid, K, H, 2 * C_hidden]
        kv_e = gather_node(kv, E_idx)

        # [*, N_rigid, H, C_hidden]
        k_e, v_e = torch.split(kv_e, self.c_hidden, dim=-1)

        # [*, N_rigid, H * P_q * 3]
        q_pts = self.linear_q_points(s)

        # This is kind of clunky, but it's how the original does it
        # [*, N_rigid, H * P_q, 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)  # [*, N_rigid, H * P_q, 3]
        # q_pts = r[..., None].apply(q_pts)
        #q_pts = loc_rigid_mul_vec(r[..., None], q_pts)  # rigid mut vec [*, N_rigid, 1] rigid * [*, N_rigid, H * P_q, 3]

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

        # rigid mut vec [*, N_rigid, 1] rigid * [*, N_rigid, H * (P_q + P_v), 3]
        #kv_pts = loc_rigid_mul_vec(r[..., None], kv_pts)

        # [*, N_rigid, H, (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))
        # [*, N_rigid, K, H, (P_q + P_v), 3]
        kv_pts_e = gather_node(kv_pts, E_idx)

        # [B, N, K, 1, 1, 3] X [B, N, K, H, (P_q + P_v), 3]
        kv_pts_e = torch.cross(
            direction[...,None,None,:],
            rot_vec(rel_ori[...,None,None,:,:], kv_pts_e)
                               )
        # [*, N_rigid, H, P_q/P_v, 3]
        k_pts_e, v_pts_e = torch.split(
            kv_pts_e, [self.no_qk_points, self.no_v_points], dim=-2
        )


        # [*, N_rigid, K, H]
        b = self.linear_b(z_e)

        # [*, H, N_rigid, K]
        qT_k = torch.matmul(
            permute_final_dims(q.unsqueeze(2), [1, 0, 2]),  # [*, H, 1, C_hidden]
            permute_final_dims(k_e, [1, 2, 0]),  # [*, H, C_hidden, K]
        ).view(q.shape[:3] + E_idx.shape[-1:]).transpose(-2, -3)

        qT_k *= math.sqrt(1.0 / (3 * self.c_hidden))  # (3 * self.c_hidden) WL * c


        b = (math.sqrt(1.0 / 3) * permute_final_dims(b, [2, 0, 1]))  # [*, H, N_rigid, K]

        a = qT_k + b

        # [*, N_rigid, K, H, P_q, 3] = [*, N_rigid, 1, *] - [*, 1, K, *]
        pt_att = q_pts.unsqueeze(-4) - k_pts_e

        pt_att = pt_att ** 2

        # [*, N_rigid, K, H, P_q]
        pt_att = sum(torch.unbind(pt_att, dim=-1))  # calculate vector length
        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        )

        pt_att = pt_att * head_weights

        # [*, N_rigid, K, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)  # Sum over point


        # [*, N_rigid, N_rigid] # MASK 随后再改
        square_mask_e = self.inf * (pair_mask - 1)  # 这里靠 mask 逼近 -inf 之后再用 softmax 让 attention score 变 0

        # [*, H, N_rigid, K]
        a = a + permute_final_dims(pt_att,[2,0,1])
        a = a + square_mask_e.unsqueeze(-3)
        a = self.softmax(a)

        ################
        # Compute output
        ################
        # [B，N，H，C] = SUM_K([B,N,H,K,1] * [B,N,H,K,C]).transpose(-2, -3)
        o = torch.sum((a[..., None] * permute_final_dims(v_e, [2,0,1,3])),dim=-2).transpose(-2, -3)

        # [*, N_rigid, H * C_hidden]
        o = flatten_final_dims(o, 2).float()

        # [*, H, 3, N_rigid, P_v]

        o_pt = torch.sum(
            (a[..., None, :, :, None] * permute_final_dims(v_pts_e,[2,4,0,1,3])),
            dim=-2).float()

        # [*, N_rigid, H, P_v, 3]
        o_pt = permute_final_dims(o_pt, [2, 0, 3, 1])
        #o_pt = loc_invert_rot_mul_vec(r[..., None, None], o_pt)

        # [*, N_rigid, H * P_v]
        o_pt_norm = flatten_final_dims(
            torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps), 2
        ).float()

        # [*, N_rigid, H * P_v, 3]
        o_pt = o_pt.reshape((*o_pt.shape[:-3], -1, 3))


        # [*, N_rigid, H, C_z] = [*, N_rigid, H, K] x [*, N_rigid, K, C_z]
        o_pair = torch.matmul(a.transpose(-2, -3), z_e.to(dtype=a.dtype))

        # [*, N_rigid, H * C_z]
        o_pair = flatten_final_dims(o_pair, 2).float()


        # [*, N_rigid, c_n]  [*, N_rigid, H * C_hidden + H * P_v * 3 + H * P_v + H * C_z]
        s = self.linear_out(
            torch.cat(
                (o.float(), *torch.unbind(o_pt.float(), dim=-1), o_pt_norm.float(), o_pair.float()), dim=-1
            )).float()

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

    def forward(self, s, rigid_mask):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)

        s = s + s_initial

        s = s * rigid_mask[...,None]
        s = self.layer_norm(s)

        return s
'''    
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

    def forward(self, node_emb, edge_emb, E_idx):
        # [batch, N_rigid, c_n/2]
        node_emb = self.initial_embed(node_emb)
        batch_size, num_rigids, _ = node_emb.shape

        # [batch, N_rigid, N_rigid, c_n]
        edge_bias = torch.cat([
            torch.tile(node_emb[:, :, None, :], (1, 1, num_rigids, 1)),
            torch.tile(node_emb[:, None, :, :], (1, num_rigids, 1, 1)),
        ], axis=-1)
        edge_bias = gather_edges(edge_bias, E_idx)

        # [batch * N_rigid * N_rigid, c_n + c_z]
        edge_emb = torch.cat([edge_emb, edge_bias], axis=-1)

        # [batch * N_rigid * N_rigid, c_z]
        edge_emb1 = self.trunk(edge_emb)
        edge_emb = self.final_layer(edge_emb1 + edge_emb)
        edge_emb = self.layer_norm(edge_emb)

        return edge_emb


class AngleScore(nn.Module):

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
        
        super(AngleScore, self).__init__()

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
        #self.linear_initial = nn.Linear(self.c_in * self.no_rigids, self.c_hidden)

        self.layers = nn.ModuleList()
        for _ in range(self.no_blocks):
            layer = AngleResnetBlock(c_hidden=self.c_hidden)
            self.layers.append(layer)

        self.linear_out = nn.Linear(self.c_hidden, self.no_angles * 2)

        self.relu = nn.ReLU()

    def forward(
        self, s: torch.Tensor
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
        #s_initial = s_initial.reshape(s_initial.shape[0], -1, s_initial.shape[-1] * self.no_rigids)
        s = s.reshape(s.shape[0], -1, s.shape[-1] * self.no_rigids)

        # [*, N_res, C_hidden]
       # s_initial = self.relu(s_initial)
        #s_initial = self.linear_initial(s_initial)
        s = self.relu(s)
        s = self.linear_in(s)
        #s = s + s_initial

        for l in self.layers:
            s = l(s)

        s = self.relu(s)

        # [*, N_res, no_angles * 2]
        s = self.linear_out(s)

        # [*, N_res, no_angles, 2]
        s = s.view(s.shape[:-1] + (-1, 2))

        unnormalized_s = s
        norm_denom = torch.sqrt(
            torch.clamp(torch.sum(s ** 2, dim=-1, keepdim=True),min=self.eps)
        )      
        s = s / norm_denom     
        return s, unnormalized_s

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
        # [B, N_rigid, 3]
        trans = self.linear(s)

        return trans

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

                 top_k
                 ):

        super(StructureUpdateModule, self).__init__()
        self.no_blocks = no_blocks
        self.top_k = top_k

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
        '''   
        self.angle_resnet = AngleResnet(c_n,
                                        c_resnet,
                                        no_resnet_blocks,
                                        no_angles,
                                        no_rigids,
                                        epsilon)
        '''
        #self.rigid_update = RigidUpdate(c_n)

        self.edge_transition = EdgeTransition(c_n,
                                              c_z,
                                              c_z)

        #self.edge_update = EdgeUpdate(pair_dim,
        #                             c_z)
        
    def forward(self,
                node_emb,
                pair_emb,
                rigids,
                direction,
                rel_ori: torch.Tensor,
                rigid_mask,
                pair_mask,
                E_idx
                ):
        B, N_rigid = rigids.shape
        sum_local_trans = torch.zeros((B,N_rigid,3),device=node_emb.device)
        modified_rigid_mask = torch.ones((B,int(N_rigid/5),5),device=node_emb.device)
        modified_rigid_mask[...,0] = 0.
        modified_rigid_mask = modified_rigid_mask.reshape(B, N_rigid) * rigid_mask
        #node_emb = torch.clone(init_node_emb)
        #node_emb = torch.clone(init_node_emb)
        for i, block in enumerate(self.blocks):
            node_emb = block(node_emb, pair_emb, rigids, direction, rel_ori, rigid_mask, pair_mask, E_idx)

            pair_emb = self.edge_transition(node_emb, pair_emb, E_idx) * pair_mask.unsqueeze(-1)

            '''
            # local translate update
            local_trans = self.rigid_update(node_emb)
            modified_local_trans = local_trans * modified_rigid_mask[...,None]
            rigids = Rigid_update_trans(rigids, modified_local_trans)
            sum_local_trans += local_trans
            '''
            # updated_chi_angles, unnormalized_chi_angles = self.angle_resnet(node_emb)

            # updated_chi_angles = torch.where(diffusion_mask[...,None], updated_chi_angles, ture_angles_sin_cos)

            # E_idx = structure_build.update_E_idx(rigids, pair_mask, self.top_k)
            
            # rigids = structure_build.torsion_to_frame(seq_idx, backbone_coords, updated_chi_angles)

           # _, _, distance, altered_direction, orientation = structure_build.frame_to_edge(rigids, seq_idx, pad_mask)

            
            # rigids = rigids * rigid_mask

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

        # [*, N_res, c_n * 5]
        """
        self.ipa = InvariantPointAttention(c_n,
                                           c_z,
                                           c_hidden,
                                           ipa_no_heads,
                                           no_qk_points,
                                           no_v_points)
        """

        self.edge_ipa = EdgeInvariantPointAttention(c_n,
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
                pair_emb,
                rigids,
                direction,
                rel_ori: torch.Tensor,
                rigid_mask,
                pair_mask,
                E_idx,
                ):

        # [*, N_rigid, c_n]
        # ipa_emb = self.ipa(node_emb, rigids,pair_mask)

        node_emb = node_emb + self.edge_ipa(node_emb, pair_emb, rigids, direction, rel_ori, pair_mask, E_idx)
        node_emb = self.ipa_ln(node_emb) # 即使是没用的rigid 最后经过layer norm之后也变的有 东西了 而且和正常的 node数值一样
        node_emb = self.node_transition(node_emb, rigid_mask)

        return node_emb

class RigidDiffusion(nn.Module):

    def __init__(self,
                 num_blocks: int = 3, # StructureUpdateModule的循环次数

                 # InputEmbedder config
                 nf_dim: int = 7 + 19 + 320,
                 c_n: int = 384, # Node channel dimension after InputEmbedding
                 relpos_k: int = 16, # relative position neighbour range
                 edge_type: int = 10,

                 # PairEmbedder parameter
                 pair_dim: int = 16 + 346 * 2 + 2*16 + 1 + 10, # rbf+3+4 + nf_dim* 2 + 2* relpos_k+1 + 10 edge type
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
                 top_k: int =64,

                 # Arc config
                 all_loc = False,
                 ):

        super(RigidDiffusion, self).__init__()

        self.all_loc = all_loc
        self.num_blocks = num_blocks
        self.top_k = top_k

        self.input_embedder = InputEmbedder(nf_dim, c_n, relpos_k, # Node feature related dim
                                            edge_type, pair_dim, c_z, # Pair feature related dim
                                            c_hidden_tri_att, c_hidden_tri_mul, # hidden dim for TriangleAttention, TriangleMultiplication
                                            pairemb_no_blocks, mha_no_heads, pair_transition_n,
                                            no_rigids)

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
                                                     epsilon,
                                                     top_k
        )

        '''
        self.noise_predictor_sincos = AngleResnet(c_n,
                                                  c_resnet,
                                                  no_resnet_blocks,
                                                  no_angles,
                                                  no_rigids,
                                                  epsilon
        )
        '''
        #self.rigid_update = RigidUpdate(c_n)

        # 预测8个 sin cos  改为预测4个角度
        self.score_predictor = AngleScore(c_n,
                                           c_resnet,
                                           no_resnet_blocks,
                                           no_angles, 
                                           no_rigids,
                                           epsilon
        )
        #self.trans_update = TransUpdate(c_n)


    def forward(self,
                rigids,
                #local_r,
                seq_idx,
                #diffusion_mask,
                sigma,
                seq_esm,
                rigid_type,
                rigid_property,
                pad_mask,
        ):
  

        # [*, N_rigid] Rigid
        # ture_angles_sin_cos = torch.stack([torch.sin(ture_angles), torch.cos(ture_angles)], dim=-1)

        # flat_mask [*, N_rigid]， others [*, N_rigid, N_rigid, c]
        pair_mask, rigid_mask, distance, altered_direction, orientation = structure_build_score.frame_to_edge(rigids, seq_idx, pad_mask)

        E_idx = structure_build_score.update_E_idx(rigids, pair_mask, self.top_k)

        altered_direction = gather_edges(altered_direction, E_idx)
        orientation = gather_edges(orientation.flatten(-2), E_idx).reshape((*E_idx.shape,3,3))
        # [*, N_rigid, c_n], [*, N_rigid, N_rigid, c_z]
        init_node_emb, pair_emb, pair_mask_e = self.input_embedder(
                                                seq_esm,
                                                #diffusion_mask,
                                                rigid_type,
                                                rigid_property,
                                                distance,
                                                #altered_direction,
                                                #orientation,
                                                rigid_mask,
                                                pair_mask,
                                                E_idx,
                                                sigma)
          
        # [B, N_rigid, c_n/3]      
        node_emb= self.structure_update(init_node_emb,
                                         pair_emb,
                                         rigids,
                                         altered_direction,
                                         orientation,
                                         rigid_mask,
                                         pair_mask_e,
                                         E_idx)

        # Reshape N_rigid into N_res 这里其实一直没有好好写， 这五个rigid的表示直接被拼起来就用来预测角度了，这里是不是应该换一种方法？
        # 直接放到 IPA 里面怎么样
        # node_emb = node_emb.reshape(node_emb.shape[0], -1, node_emb.shape[-1] * 5)

        # [B, N_rigid, c_n] -->  [B, N_res,4]    
        score = self.score_predictor(node_emb, init_node_emb)
        #local_trans = self.rigid_update(node_emb)
        #transed_local_r = self.trans_update(node_emb, local_r)

        # [B, N_res,4],  [B, N_rigid, 3] 
        return score#, local_trans


