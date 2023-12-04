import math
from abc import ABC

import torch
from torch import Tensor
from torch.nn import Parameter, Softplus
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import softmax

from write_preds_pdb.geometry import Rigid, Rotation, from_tensor_4x4
from write_preds_pdb.geometry import loc_invert_rot_mul_vec, loc_rigid_mul_vec
from model.utils1 import ipa_point_weights_init_

class GraphIPA(MessagePassing, ABC):

    def __init__(self,
                 c_n: int,
                 c_hidden: int,
                 c_z: int,
                 heads: int = 8,
                 no_qk_points: int = 4,
                 no_v_points: int = 8,
                 add_self_loops: bool = False,
                 inf = 1e8,
                 eps: float = 1e-6):

        super(GraphIPA, self).__init__()

        self.node_dim = 0
        self.heads = heads
        self.c_hidden = c_hidden
        #self.concat = concat
        #self.negative_slope = negative_slope
        #self.dropout = dropout
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.add_self_loops = add_self_loops
        self.edge_out_channels = c_z
        #self.fill_value = fill_value
        self.inf = inf
        self.eps = eps

        hc = heads * c_hidden
        self.lin_src_kv = Linear(c_n, 2*hc, bias= False)
        self.lin_dst_q = Linear(c_n, hc, bias=False)

        hpq = self.heads * self.no_qk_points * 3
        self.lin_dst_q_points = Linear(c_n, hpq, bias=False)

        hpkv = self.heads * (self.no_qk_points + self.no_v_points) * 3
        self.lin_src_kv_points = Linear(c_n, hpkv, bias=False)

        self.linear_b = Linear(c_z, heads, bias=False)

        concat_out_dim = self.heads * (
                c_z + c_hidden + self.no_v_points * 4
        )
        self.linear_out = Linear(concat_out_dim, c_n)

        self.head_weights = Parameter(torch.zeros(self.heads))
        ipa_point_weights_init_(self.head_weights)
        self.softplus = Softplus()


    def forward(self, x, edge_attr, data) -> Tensor:

        '''
        x: node_emb [N, C_n]
        edge_index: [2, N_e]
        edge_attr: [E, C_e]
        r: [N_batch] Rigid
        '''

        edge_index = data.edge_index
        n_edge = data.num_edges
        
        r = from_tensor_4x4(data.rigid.to(x.device))

        # [N, 2* H * C_hidden]
        src_kv = self.lin_src_kv(x)
        # [N, H * C_hidden]
        dst_q = self.lin_dst_q(x)

        # [N, H, C_hidden]
        dst_q = dst_q.view(-1, self.heads, self.c_hidden)

        # [N, H, 2 * C_hidden]
        src_kv = src_kv.view(-1, self.heads, 2 * self.c_hidden)
        # [N, H, C_hidden]
        src_k, src_v = torch.split(src_kv, self.c_hidden, dim=-1)

        # [N, H * P_q * 3]
        dst_q_pts = self.lin_dst_q_points(x)
        # [N, H * P_q, 3]
        dst_q_pts = torch.split(dst_q_pts, dst_q_pts.shape[-1] // 3, dim=-1)
        dst_q_pts = torch.stack(dst_q_pts, dim=-1)
        dst_q_pts = loc_rigid_mul_vec(r[..., None], dst_q_pts)
        # [N, H, P_q, 3]
        dst_q_pts = dst_q_pts.view(-1, self.heads, self.no_qk_points, 3)

        # [N, H * (P_q +P_v) * 3]
        src_kv_pts = self.lin_src_kv_points(x)
        # [N, H * (P_q +P_v), 3]
        src_kv_pts = torch.split(src_kv_pts, src_kv_pts.shape[-1] // 3, dim=-1)
        src_kv_pts = torch.stack(src_kv_pts, dim=-1)
        src_kv_pts = loc_rigid_mul_vec(r[..., None], src_kv_pts)

        # [N, H, (P_q +P_v), 3]
        src_kv_pts = src_kv_pts.view(-1, self.heads, self.no_v_points + self.no_qk_points, 3)

        src_k_pts, src_v_pts = torch.split(src_kv_pts, [self.no_qk_points, self.no_v_points], dim=-2)
        # [E, H]
        b = self.linear_b(edge_attr)

        alpha = (src_k, dst_q)
        alpha_pts = (src_k_pts, dst_q_pts)

        alpha = self.edge_updater(edge_index,
                                  alpha=alpha,
                                  alpha_pts = alpha_pts,
                                  edge_attr=b)

        x = (src_v, x)
        # [N, H, (P_q/P_v), 3]
        v = (src_v_pts, dst_q_pts)

        # [N, H * Pv * 3 + C_hidden + C_e]
        out = self.propagate(edge_index, x=x, v=v, rigid=r[edge_index[0]], edge_attr = edge_attr, alpha=alpha, n_edge=n_edge)
        o_pt = out[:,:self.heads*self.no_v_points*3]
        o_pt = o_pt.reshape(data.num_nodes,-1,3)
        # [N, H * Pv * 3 + C_hidden + C_e]
        o_pt_norm = torch.sqrt(torch.sum(o_pt ** 2, dim=-1)+ self.eps)
        out = torch.cat((out, o_pt_norm),dim=-1).float()

        out = self.linear_out(out)

        return out

    def edge_update(self,
                    alpha_j,
                    alpha_i,
                    alpha_pts_i,
                    alpha_pts_j,
                    edge_attr, index, size_i) -> Tensor:

        # [E, H, 1, 1] -> [E, H]
        qT_k = torch.matmul(alpha_i.unsqueeze(2), alpha_j.unsqueeze(3)).view(-1, self.heads)

        qT_k *= math.sqrt(1.0 / (3 * self.c_hidden))
        b = math.sqrt(1.0 / 3) * edge_attr

        alpha = qT_k + b

        # [E, H, P_q]
        pt_att = sum(torch.unbind((alpha_pts_i - alpha_pts_j)**2, dim=-1))

        head_weights = self.softplus(self.head_weights).view(1, -1, 1)
        head_weights = head_weights * math.sqrt(1.0 / (3 * (self.no_qk_points * 9.0 / 2)))

        pt_att = pt_att * head_weights

        # [E, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)

        alpha = alpha + pt_att
        alpha = softmax(alpha, index)

        return alpha

    def message(self,
                x_j: Tensor, # [E, H, C_hidden]
                v_j: Tensor,  # [E, H, P_v, 3]
                rigid_i: Tensor,  # [E,]
                edge_attr: Tensor, # [E, C_e]
                alpha: Tensor, # [E, H]
                n_edge
                ) -> Tensor:

        # [E, H, C_hidden]
        o = x_j * alpha[..., None]
        o = o.reshape(n_edge, -1)
        # [E, H, C_e]
        o_pair = edge_attr[..., None,:] * alpha[..., None]
        o_pair = o_pair.reshape(n_edge, -1)
        # [E, H, P_v, 3]
        o_pt = v_j * alpha[..., None, None]
        o_pt = loc_invert_rot_mul_vec(rigid_i[..., None, None], o_pt)
        o_pt = o_pt.reshape(n_edge, -1, 3)

        # [E, H * Pv * 3 + C_hidden + C_e]
        return torch.cat((*torch.unbind(o_pt, dim=-1), o, o_pair), dim=-1)
