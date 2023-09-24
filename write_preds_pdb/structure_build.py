import torch
import torch.nn as nn
import numpy as np
from constant import (restype_rigid_group_default_frame,
                      restype_atom14_to_rigid_group,
                      restype_atom14_mask,
                      restype_atom14_rigid_group_positions,
                      restype_atom37_mask,
                      make_atom14_37_list,
                      restype_frame_mask
                      )
import geometry
import protein
import os


def rotate_sidechain(
                    restype_idx:torch.Tensor, # [*, N]
                    angles: torch.Tensor # [*,N,4，2]
                    ) -> geometry.Rigid:

    # [21, 8, 4, 4]
    default_frame = torch.tensor(restype_rigid_group_default_frame,
                              dtype=angles.dtype,
                              device=angles.device,
                              requires_grad=False)
    # [*, N, 8, 4, 4]
    res_default_frame = default_frame[restype_idx, ...]
    
    #print(" res_default_frame", res_default_frame.shape)
    # [*, N, 8] Rigid
    default_r = geometry.from_tensor_4x4(res_default_frame)

    sin_angles = angles[..., 0] # [*,N,4]
    cos_angles = angles[..., 1]

    # [*,N,4] + [*,N,4] == [*,N,8]
    # adding 4 zero angles which means no change to the default value.
    #=============================训练时.to('cpu')保留，解开注释===================================================#
    sin_angles = torch.cat([torch.zeros(*restype_idx.shape, 4).to('cpu'), sin_angles.to('cpu')],dim=-1)
    cos_angles = torch.cat([torch.ones(*restype_idx.shape, 4).to('cpu'), cos_angles.to('cpu')],dim=-1)
    #=============================训练时.to('cpu')保留，解开注释===================================================#

    
    #=============================训练时.to('cpu')移除，解开注释============================================================#
    #sin_angles = torch.cat([torch.zeros(*restype_idx.shape, 4), sin_angles],dim=-1)
    #cos_angles = torch.cat([torch.ones(*restype_idx.shape, 4), cos_angles],dim=-1)
    #=============================训练时.to('cpu')移除，解开注释============================================================#

    #print("sin_angles==",sin_angles.shape)
    
    # [*, N, 8, 3, 3]
    # Produces rotation matrices of the form:
    # [
    #   [1, 0  , 0  ],
    #   [0, a_2,-a_1],
    #   [0, a_1, a_2]
    # ]
    # This follows the original code rather than the supplement, which uses
    # different indices.
    all_rots = angles.new_zeros(default_r.rot.get_rot_mat().shape)
    #print("orign all_rots==",all_rots.shape)
    all_rots[..., 0, 0] = 1
    all_rots[..., 1, 1] = cos_angles
    all_rots[..., 1, 2] = -sin_angles
    all_rots[..., 2, 1] = sin_angles
    all_rots[..., 2, 2] = cos_angles
    
    #print("all_rots==",all_rots.shape) # torch.Size([128, 8, 3, 3])
    #print('Rotation =========',geometry.Rotation(rot_mats = all_rots).shape)
    all_rots = geometry.Rigid(geometry.Rotation(rot_mats = all_rots), None)
    
    #print("final all_rots==",all_rots.shape) #torch.Size([128])
    
    #print("default_r==",default_r.shape) #torch.Size([128, 8])
    all_frames = geometry.Rigid_mult(default_r,all_rots)

    # Rigid
    chi2_frame_to_frame = all_frames[..., 5]
    chi3_frame_to_frame = all_frames[..., 6]
    chi4_frame_to_frame = all_frames[..., 7]

    chi1_frame_to_bb = all_frames[..., 4]
    chi2_frame_to_bb = geometry.Rigid_mult(chi1_frame_to_bb, chi2_frame_to_frame)
    chi3_frame_to_bb = geometry.Rigid_mult(chi2_frame_to_bb, chi3_frame_to_frame)
    chi4_frame_to_bb = geometry.Rigid_mult(chi3_frame_to_bb, chi4_frame_to_frame)

    all_frames_to_bb = geometry.cat(
        [all_frames[..., :5],
        chi2_frame_to_bb.unsqueeze(-1),
        chi3_frame_to_bb.unsqueeze(-1),
        chi4_frame_to_bb.unsqueeze(-1),],
        dim=-1,
    )

    return all_frames_to_bb

def frame_to_pos(frames, aatype_idx):

    # [21 , 14]
    group_index = torch.tensor(restype_atom14_to_rigid_group)

    # [21 , 14] idx [*, N] -> [*, N, 14]
    group_mask = group_index[aatype_idx, ...]
    # [*, N, 14, 8]
    group_mask = nn.functional.one_hot(group_mask, num_classes = frames.shape[-1])

    # [*, N, 14, 8] Rigid frames for every 14 atoms, non exist atom are mapped to group 0
    map_atoms_to_global = frames[..., None, :] * group_mask # [*, N, :, 8] * [*, N, 14, 8]

    # [*, N, 14]
    map_atoms_to_global = geometry.map_rigid_fn(map_atoms_to_global)

    # [21 , 14]
    atom_mask = torch.tensor(restype_atom14_mask)
    # [*, N, 14, 1]
    atom_mask = atom_mask[aatype_idx, ...].unsqueeze(-1)

    # [21, 14, 3]
    default_pos = torch.tensor(restype_atom14_rigid_group_positions)
    # [*, N, 14, 3]
    default_pos = default_pos[aatype_idx, ...]

    pred_pos = geometry.rigid_mul_vec(map_atoms_to_global, default_pos)
    pred_pos = pred_pos * atom_mask

    return pred_pos

def batch_gather(data,  # [N, 14, 3]
                 indexing): # [N,37]
    ranges = []

    N = data.shape[-3]
    r = torch.arange(N)
   # print("r1========",r.shape)
    r = r.view(-1,1)
   # print("r2========",r.shape)
    ranges.append(r)
   # print("r3========",r.shape)
    remaining_dims = [slice(None) for _ in range(2)]
   # print("remaining_dims1========",remaining_dims.shape)
    remaining_dims[-2] = indexing
   # print("remaining_dims2========",remaining_dims.shape)
    ranges.extend(remaining_dims)# [Tensor(N,1), Tensor(N,37), slice(None)]
   # print("ranges========",ranges.shape)
    return data[ranges] # [N, 37, 3]

def atom14_to_atom37(atom14, aa_idx): # atom14: [*, N, 14, 3]
    
    restype_atom37_to_atom14 = make_atom14_37_list() #注意有错
    
    residx_atom37_to_14 = restype_atom37_to_atom14[aa_idx]
    # [N, 37]
    atom37_mask = restype_atom37_mask[aa_idx]

    # [N, 37, 3]
   # print('atom14===========', atom14.shape)
   # print('atom37_mask===========', atom37_mask.shape)
   # print('residx_atom37_to_14=====', residx_atom37_to_14.shape)
    atom37 = batch_gather(atom14, residx_atom37_to_14)
    atom37 = atom37 * atom37_mask[...,None]

    return atom37

def torsion_to_position(aatype_idx: torch.Tensor, # [*, N]
                        backbone_position: torch.Tensor, # [*, N, 4, 3] (N, CA, C, O)
                        angles: torch.Tensor, # [*, N, 4] (X1, X2, X3, X4)
                        ): # -> [*, N, 14, 3]
    """Compute Side Chain Atom position using the predicted torsion
    angle and the fixed backbone coordinates.

    Args:
        aatype_idx: aatype for each residue
        backbone_position: backbone coordinate for each residue
        angles: torsion angles for each residue

    return:
        all atom position [N, X] X are # number of atoms (14?
    """

    # side chain frames [*, N, 8] Rigid
    angles_sin_cos = torch.stack(
            [
                torch.sin(angles),
                torch.cos(angles),
            ],
            dim=-1,
        )
    sc_to_bb = rotate_sidechain(aatype_idx, angles_sin_cos)
  #  print('sc_to_bb ==========================',sc_to_bb.shape)
    # [*, N] Rigid
   # print('backbone_position =================', backbone_position.shape)
    bb_to_gb = geometry.get_gb_trans(backbone_position)

    ''''
    bb_to_gb = torch.tensor(
                            [make_rigid_trans(
                            ex = res[2] - res[1], # C-CA,
                            y_vec = res[0] - res[1], # N-CA
                            t = res[1])  for res in backbone_position]
    ) # [N, 4, 4]
    '''

    all_frames_to_global = geometry.Rigid_mult(bb_to_gb[..., None], sc_to_bb)
#    print('all_frames_to_global==============', all_frames_to_global.shape)
    # [*, N, 14, 3]
    all_pos = frame_to_pos(all_frames_to_global, aatype_idx)
   # print('all_pose=============', all_pos.shape)
    # [*, N, 37, 3]
 #   print('aaidx type ===================== ',type(aatype_idx))
#    print('aaidx ===================== ',aatype_idx)
    final_pos = atom14_to_atom37(all_pos, aatype_idx)

    return final_pos

def torsion_to_frame(aatype_idx: torch.Tensor, # [*, N]
                    backbone_position: torch.Tensor, # [*, N, 4, 3] (N, CA, C, O)
                    angles_sin_cos: torch.Tensor, # [*, N, 4, 2] (X1, X2, X3, X4)
                    ): # -> [*, N, 5] Rigid
    """Compute all residue frames given torsion
        angles and the fixed backbone coordinates.

        Args:
            aatype_idx: aatype for each residue
            backbone_position: backbone coordinate for each residue
            angles: torsion angles for each residue

        return:
            all frames [N, 5] Rigid
        """

    # side chain frames [*, N, 5] Rigid
    # We create 3 dummy identity matrix for omega and other angles which is not used in the frame attention process
    sc_to_bb = rotate_sidechain(aatype_idx, angles_sin_cos)[..., [0,4,5,6,7]]

    # [*, N_res] Rigid
    bb_to_gb = geometry.get_gb_trans(backbone_position)

    all_frames_to_global = geometry.Rigid_mult(bb_to_gb[..., None], sc_to_bb)

    # [*, N_rigid]
    flatten_frame = geometry.flatten_rigid(all_frames_to_global)

    return flatten_frame # return frame

def frame_to_edge(frames: geometry.Rigid, # [*, N_rigid] Rigid
                  aatype_idx, # [*, N_res]
                  pad_mask # [*, N_res]
                  ):
    """
    compute edge information between two frames distance, direction, orientation
    Args:
        frames: protein rigid frames [*, N_, 5]

    Returns:

    """

    # [20, 5]
    restype_frame5_mask = torch.tensor(restype_frame_mask)

    # [*, N_res, 5]
    frame_mask = restype_frame5_mask[aatype_idx, ...].to('cpu')
    frame_mask = frame_mask * pad_mask[..., None]
    
    # [*, N_rigid]
    flat_mask= torch.flatten(frame_mask, start_dim= -2)

    # [*, N_rigid, N_rigid]
    pair_mask = flat_mask[..., None] * flat_mask[..., None, :]
    # [*, N_rigid, N_rigid]
    distance, altered_direction, orientation = frames.edge()

    return pair_mask, flat_mask, distance, altered_direction, orientation


def update_E_idx(frames: geometry.Rigid,  # [*, N_rigid] Rigid
                pair_mask: torch.Tensor,  # [*, N_res]
                top_k: int,
                ):


    # [*, N_rigid, N_rigid]
    distance, _, _ = frames.edge()

    D = pair_mask * distance

    D_max, _ = torch.max(D, -1, keepdim=True)
    D_adjust = D + (1. - pair_mask) * D_max  # give masked position value D_max

    # Value of distance [*, N_rigid, K], Index of distance [*, N_rigid, K]
    _, E_idx = torch.topk(D_adjust, top_k, dim=-1, largest=False)

    return E_idx


def write_preds_pdb_file(structure, sampled_dfs, out_path, fname, j):
    

    final_atom_mask = restype_atom37_mask[structure["seq"]]

    seq_list = torch.from_numpy(structure["seq"])
    coord_list = structure["coords"]
    
    coord_list = torch.from_numpy(coord_list)
    angle_list = torch.from_numpy(sampled_dfs)
    final_atom_positions = torsion_to_position(seq_list, 
                                               coord_list,
                                                angle_list) 
    
    chain_len = len(seq_list)
    index = np.arange(1,chain_len+1)
    resulted_protein = protein.Protein(
                            aatype=structure["seq"], # [*,N]
                            atom_positions=final_atom_positions,
                            atom_mask=final_atom_mask,
                            residue_index=index, #0,1,2,3,4 range_chain
                            b_factors=np.zeros_like(final_atom_mask))
    
    pdb_str = protein.to_pdb(resulted_protein) 
    
    with open(os.path.join(out_path,f"{fname}_generate_{j}.pdb"), 'w') as fp:
         fp.write(pdb_str)
         