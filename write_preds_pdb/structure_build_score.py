import torch
import torch.nn as nn
import numpy as np
from write_preds_pdb.constant import (restype_rigid_group_default_frame,
                      restype_atom14_to_rigid_group,
                      restype_atom14_mask,
                      restype_atom14_rigid_group_positions,
                      restype_atom37_mask,
                      make_atom14_37_list,
                      restype_frame_mask,
                      end_rigid,
                      )

import write_preds_pdb.constant as rc
import write_preds_pdb.geometry as geometry
import write_preds_pdb.protein as protein
import os

device1 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def rotate_sidechain(
        restype_idx: torch.Tensor,  # [*, N]
        angles: torch.Tensor,  # [*,N,4，2]
        last_local_r: geometry.Rigid,  # [*, N, 8] Rigid
) -> geometry.Rigid:
    
    sin_angles = torch.sin(angles) 
    cos_angles = torch.cos(angles)

    # [*,N,4] + [*,N,4] == [*,N,8]
    # adding 4 zero angles which means no change to the default value.
    sin_angles = torch.cat([torch.zeros(*restype_idx.shape, 4).to(sin_angles.device), sin_angles], dim=-1)
    cos_angles = torch.cat([torch.ones(*restype_idx.shape, 4).to(sin_angles.device), cos_angles], dim=-1)

    # [*, N, 8, 3, 3]
    # Produces rotation matrices of the form:
    # [
    #   [1, 0  , 0  ],
    #   [0, a_2,-a_1],
    #   [0, a_1, a_2]
    # ]
    # This follows the original code rather than the supplement, which uses
    # different indices.
    if type(last_local_r) == list:
        rot = torch.cat([rigid.rot.get_rot_mat() for rigid in last_local_r], 0)
        trans = torch.cat([rigid.trans for rigid in last_local_r], 0)
        loc = torch.cat([rigid.loc for rigid in last_local_r], 0)
        last_local_r = geometry.Rigid(geometry.Rotation(rot), trans, loc).cuda()

    all_rots = sin_angles.new_zeros(last_local_r.rot.get_rot_mat().shape).to(sin_angles.device)
    # print("orign all_rots==",all_rots.shape)
    all_rots[..., 0, 0] = 1
    all_rots[..., 1, 1] = cos_angles
    all_rots[..., 1, 2] = -sin_angles
    all_rots[..., 2, 1] = sin_angles
    all_rots[..., 2, 2] = cos_angles

    all_rots = geometry.Rigid(geometry.Rotation(rot_mats=all_rots), None, None)

    all_frames = geometry.Rigid_mult(last_local_r, all_rots)

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
         chi4_frame_to_bb.unsqueeze(-1), ],
        dim=-1,
    )

    return all_frames_to_bb, all_frames


def frame_to_pos(frames, aatype_idx, bb_cords):
    # [21 , 14]
    group_index = torch.tensor(restype_atom14_to_rigid_group).to('cuda')

    # [21 , 14] idx [*, N] -> [*, N, 14]
    group_mask = group_index[aatype_idx, ...]
    # [*, N, 14, 8]
    group_mask = nn.functional.one_hot(group_mask, num_classes=frames.shape[-1])

    # [*, N, 14, 8] Rigid frames for every 14 atoms, non exist atom are mapped to group 0
    map_atoms_to_global = frames[..., None, :] * group_mask  # [*, N, :, 8] * [*, N, 14, 8]

    # [*, N, 14]
    map_atoms_to_global = geometry.map_rigid_fn(map_atoms_to_global)

    # [21 , 14]
    atom_mask = torch.tensor(restype_atom14_mask).to('cuda')
    # [*, N, 14, 1]
    atom_mask = atom_mask[aatype_idx, ...].unsqueeze(-1)

    # [21, 14, 3]
    default_pos = torch.tensor(restype_atom14_rigid_group_positions).to('cuda')
    # [*, N, 14, 3]
    default_pos = default_pos[aatype_idx, ...]

    pred_pos = geometry.rigid_mul_vec(map_atoms_to_global, default_pos)
    pred_pos = pred_pos * atom_mask

    pred_pos, _ = atom14_to_atom37_batched(pred_pos, aatype_idx)

    # this is just for convinient use of the backbone coordinate
    # [B, N, 37, 3] [B,N,4,3]
    pred_pos[...,:3,:] = bb_cords[...,:3,:]
    pred_pos[...,4,:] = bb_cords[...,3,:]

    return pred_pos


def batched_gather(data, inds, dim=0, no_batch_dims=0):
    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s)  # torch.arange N
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))  # [N, 1]
        ranges.append(r)

    remaining_dims = [
        slice(None) for _ in range(len(data.shape) - no_batch_dims)
    ]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)  # [Tensor(N,1), Tensor(N,37), slice(None)]
    return data[ranges]  # [N, 37, 3]


def atom14_to_atom37_batched(atom14, aa_idx):  # atom14: [*, N, 14, 3]

    restype_atom37_to_atom14 = make_atom14_37_list()  # 注意有错

    residx_atom37_to_14 = restype_atom37_to_atom14[aa_idx]

    # [N, 37]
    atom37_mask = torch.tensor(restype_atom37_mask).to('cuda')
    atom37_mask = atom37_mask[aa_idx]

    # [N, 37, 3]
    atom37 = batched_gather(atom14,
                            residx_atom37_to_14,
                            dim=-2,
                            no_batch_dims=len(atom14.shape[:-2])
                            )
    atom37 = atom37 * atom37_mask[..., None]

    return atom37, atom37_mask


def batch_gather(data,  # [N, 14, 3]
                 indexing):  # [N,37]
    ranges = []

    N = data.shape[-3]
    r = torch.arange(N)
    # print("r1========",r.shape)
    r = r.view(-1, 1)
    # print("r2========",r.shape)
    ranges.append(r)
    # print("r3========",r.shape)
    remaining_dims = [slice(None) for _ in range(2)]
    # print("remaining_dims1========",remaining_dims.shape)
    remaining_dims[-2] = indexing
    # print("remaining_dims2========",remaining_dims.shape)
    ranges.extend(remaining_dims)  # [Tensor(N,1), Tensor(N,37), slice(None)]
    # print("ranges========",ranges.shape)
    return data[ranges]  # [N, 37, 3]


def atom14_to_atom37(atom14, aa_idx):  # atom14: [*, N, 14, 3]

    restype_atom37_to_atom14 = make_atom14_37_list()  # 注意有错

    residx_atom37_to_14 = restype_atom37_to_atom14[aa_idx]
    # [N, 37]
    atom37_mask = torch.tensor(restype_atom37_mask)
    atom37_mask = atom37_mask[aa_idx]

    # [N, 37, 3]
    # print('atom14===========', atom14.shape)
    # print('atom37_mask===========', atom37_mask.shape)
    # print('residx_atom37_to_14=====', residx_atom37_to_14.shape)
    atom37 = batch_gather(atom14, residx_atom37_to_14)
    atom37 = atom37 * atom37_mask[..., None]

    return atom37


def torsion_to_position(aatype_idx: torch.Tensor,  # [*, N]
                        backbone_position: torch.Tensor,  # [*, N, 4, 3] (N, CA, C, O)
                        angles: torch.Tensor,  # [*, N, 4] (X1, X2, X3, X4)
                        last_step_r,
                        ):  # -> [*, N, 14, 3]
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
    sc_to_bb, current_r = rotate_sidechain(aatype_idx, angles_sin_cos, last_step_r)
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
    all_pos[..., :4, :] = backbone_position

    # print('all_pose=============', all_pos.shape)
    # [*, N, 37, 3]
    #   print('aaidx type ===================== ',type(aatype_idx))
    #    print('aaidx ===================== ',aatype_idx)
    final_pos = atom14_to_atom37(all_pos, aatype_idx)

    return final_pos


def get_default_r(restype_idx):
    default_frame = torch.tensor(restype_rigid_group_default_frame)

    # [*, N, 8, 4, 4]
    res_default_frame = default_frame[restype_idx, ...]

    # [*, N, 8] Rigid
    default_r = geometry.from_tensor_4x4(res_default_frame)
    return default_r


def torsion_to_frame(angles,
                     protein
                     ):  # -> [*, N, 5] Rigid
    """Compute all residue frames given torsion
        angles and the fixed backbone coordinates.

        Args:
            aatype_idx: aatype for each residue
            backbone_position: backbone coordinate for each residue
            angles: torsion angles for each residue

        return:
            all frames [N, 5] Rigid
        """

    bb_to_gb = geometry.get_gb_trans(protein.bb_coord)
    
    sc_to_bb, local_r = rotate_sidechain(protein.aatype, angles, protein.local_rigid)
    all_frames_to_global = geometry.Rigid_mult(bb_to_gb[..., None], sc_to_bb)

    # [N_rigid] Rigid
    flatten_frame = geometry.flatten_rigid(all_frames_to_global[..., [0, 4, 5, 6, 7]])

    flat_rigids = flatten_frame[protein.rigid_mask]

    return flat_rigids, local_r, all_frames_to_global




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


def write_pdb_from_position(graph, final_atom_positions, out_path, fname, j):
    final_atom_mask = restype_atom37_mask[graph.aatype.cpu()]
    chain_len = len(graph.aatype)
    index = np.arange(1, chain_len + 1)

    resulted_protein = protein.Protein(
        aatype=graph.aatype.cpu(),  # [*,N]
        atom_positions=final_atom_positions,
        atom_mask=final_atom_mask,
        residue_index=index,  # 0,1,2,3,4 range_chain
        b_factors=np.zeros_like(final_atom_mask))

    pdb_str = protein.to_pdb(resulted_protein)

    with open(os.path.join(out_path, f"{fname}_generate_{j}.pdb"), 'w') as fp:
        fp.write(pdb_str)


def write_preds_pdb_file(structure, sampled_dfs, out_path, fname, j):
    final_atom_mask = restype_atom37_mask[structure["seq"]]

    seq_list = torch.from_numpy(structure["seq"])
    coord_list = structure["coords"]

    coord_list = torch.from_numpy(coord_list)
    angle_list = torch.from_numpy(sampled_dfs)
    default_r = get_default_r(seq_list, angle_list)
    final_atom_positions = torsion_to_position(seq_list,
                                               coord_list,
                                               angle_list,
                                               default_r)

    chain_len = len(seq_list)
    index = np.arange(1, chain_len + 1)
    resulted_protein = protein.Protein(
        aatype=structure["seq"],  # [*,N]
        atom_positions=final_atom_positions,
        atom_mask=final_atom_mask,
        residue_index=index,  # 0,1,2,3,4 range_chain
        b_factors=np.zeros_like(final_atom_mask))

    pdb_str = protein.to_pdb(resulted_protein)

    with open(os.path.join(out_path, f"{fname}_generate_{j}.pdb"), 'w') as fp:
        fp.write(pdb_str)


def get_chi_atom_indices():
    """Returns atom indices needed to compute chi angles for all residue types.

    Returns:
      A tensor of shape [residue_types=21, chis=4, atoms=4]. The residue types are
      in the order specified in rc.restypes + unknown residue type
      at the end. For chi angles which are not defined on the residue, the
      positions indices are by default set to 0.
    """
    chi_atom_indices = []
    for residue_name in rc.restypes:
        residue_name = rc.restype_1to3[residue_name]
        residue_chi_angles = rc.chi_angles_atoms[residue_name]
        atom_indices = []
        for chi_angle in residue_chi_angles:
            atom_indices.append([rc.atom_order[atom] for atom in chi_angle])
        for _ in range(4 - len(atom_indices)):
            atom_indices.append(
                [0, 0, 0, 0]
            )  # For chi angles not defined on the AA.
        chi_atom_indices.append(atom_indices)

    chi_atom_indices.append([[0, 0, 0, 0]] * 4)  # For UNKNOWN residue.

    return chi_atom_indices


def batched_gather(data, inds, dim=0, no_batch_dims=0):
    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s)  # torch.arange N
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))  # [N, 1]
        ranges.append(r)

    remaining_dims = [
        slice(None) for _ in range(len(data.shape) - no_batch_dims)
    ]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)  # [Tensor(N,1), Tensor(N,37), slice(None)]
    return data[ranges]  # [N, 37, 3]


def rigids_to_torsion_angles(
        aatype_idx,
        rigids,
):
    """
    Convert coordinates to torsion angles.

    This function is extremely sensitive to floating point imprecisions
    and should be run with double precision whenever possible.

    Args:
        Dict containing:
            * (prefix)aatype:
                [*, N_res] residue indices
            * (prefix)all_atom_positions:
                [*, N_res, 37, 3] atom positions (in atom37
                format)
            * (prefix)all_atom_mask:
                [*, N_res, 37] atom position mask
    Returns:
        The same dictionary updated with the following features:

        "(prefix)torsion_angles_sin_cos" ([*, N_res, 7, 2])
            Torsion angles
        "(prefix)alt_torsion_angles_sin_cos" ([*, N_res, 7, 2])
            Alternate torsion angles (accounting for 180-degree symmetry)
        "(prefix)torsion_angles_mask" ([*, N_res, 7])
            Torsion angles mask
    """
    all_pos = frame_to_pos(rigids, aatype_idx)
    # [*, N, 37, 3], [*, N_res, 37] atom position mask
    all_atom_positions, all_atom_mask = atom14_to_atom37_batched(all_pos, aatype_idx)

    # Returns the same result as torsion_to_position but Batched all_atom_positions

    aatype = torch.clamp(aatype_idx, max=20)

    pad = all_atom_positions.new_zeros(
        [*all_atom_positions.shape[:-3], 1, 37, 3]
    )
    prev_all_atom_positions = torch.cat(
        [pad, all_atom_positions[..., :-1, :, :]], dim=-3
    )

    pre_omega_atom_pos = torch.cat(
        [prev_all_atom_positions[..., 1:3, :], all_atom_positions[..., :2, :]],
        dim=-2,
    )
    phi_atom_pos = torch.cat(
        [prev_all_atom_positions[..., 2:3, :], all_atom_positions[..., :3, :]],
        dim=-2,
    )
    psi_atom_pos = torch.cat(
        [all_atom_positions[..., :3, :], all_atom_positions[..., 4:5, :]],
        dim=-2,
    )

    chi_atom_indices = torch.as_tensor(
        get_chi_atom_indices(), device=aatype.device
    )

    atom_indices = chi_atom_indices[..., aatype, :, :]
    chis_atom_pos = batched_gather(
        all_atom_positions, atom_indices, -2, len(atom_indices.shape[:-2])
    )

    torsions_atom_pos = torch.cat(
        [
            pre_omega_atom_pos[..., None, :, :],
            phi_atom_pos[..., None, :, :],
            psi_atom_pos[..., None, :, :],
            chis_atom_pos,
        ],
        dim=-3,
    )

    torsion_frames = geometry.Rigid.from_3_points(
        torsions_atom_pos[..., 1, :],
        torsions_atom_pos[..., 2, :],
        torsions_atom_pos[..., 0, :],
        eps=1e-8,
    )

    fourth_atom_rel_pos = geometry.invert_rot_mul_vec(torsion_frames, torsions_atom_pos[..., 3, :])

    torsion_angles_sin_cos = torch.stack(
        [fourth_atom_rel_pos[..., 2], fourth_atom_rel_pos[..., 1]], dim=-1
    )

    denom = torch.sqrt(
        torch.sum(
            torch.square(torsion_angles_sin_cos),
            dim=-1,
            dtype=torsion_angles_sin_cos.dtype,
            keepdims=True,
        )
        + 1e-8
    )
    torsion_angles_sin_cos = torsion_angles_sin_cos / denom

    torsion_angles_sin_cos = torsion_angles_sin_cos * all_atom_mask.new_tensor(
        [1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
    )[((None,) * len(torsion_angles_sin_cos.shape[:-2])) + (slice(None), None)]

    return all_atom_positions


def atom37_to_torsion_feature(
        aatype,
        all_atom_positions,
        all_atom_mask,
):
    """
    Convert coordinates to torsion angles.

    This function is extremely sensitive to floating point imprecisions
    and should be run with double precision whenever possible.

    Args:
        Dict containing:
            * (prefix)aatype:
                [*, N_res] residue indices
            * (prefix)all_atom_positions:
                [*, N_res, 37, 3] atom positions (in atom37
                format)
            * (prefix)all_atom_mask:
                [*, N_res, 37] atom position mask
    Returns:
        The same dictionary updated with the following features:

        "(prefix)torsion_angles_sin_cos" ([*, N_res, 7, 2])
            Torsion angles
        "(prefix)alt_torsion_angles_sin_cos" ([*, N_res, 7, 2])
            Alternate torsion angles (accounting for 180-degree symmetry)
        "(prefix)torsion_angles_mask" ([*, N_res, 7])
            Torsion angles mask
    """

    feat = {}
    aatype = torch.clamp(aatype, max=20)
    all_atom_positions = torch.tensor(all_atom_positions)
    all_atom_mask = torch.tensor(all_atom_mask)

    aatype = torch.clamp(aatype, max=20)

    pad = all_atom_positions.new_zeros(
        [*all_atom_positions.shape[:-3], 1, 37, 3]
    )
    prev_all_atom_positions = torch.cat(
        [pad, all_atom_positions[..., :-1, :, :]], dim=-3
    )

    pad = all_atom_mask.new_zeros([*all_atom_mask.shape[:-2], 1, 37])
    prev_all_atom_mask = torch.cat([pad, all_atom_mask[..., :-1, :]], dim=-2)

    pre_omega_atom_pos = torch.cat(
        [prev_all_atom_positions[..., 1:3, :], all_atom_positions[..., :2, :]],
        dim=-2,
    )
    phi_atom_pos = torch.cat(
        [prev_all_atom_positions[..., 2:3, :], all_atom_positions[..., :3, :]],
        dim=-2,
    )
    psi_atom_pos = torch.cat(
        [all_atom_positions[..., :3, :], all_atom_positions[..., 4:5, :]],
        dim=-2,
    )

    pre_omega_mask = torch.prod(
        prev_all_atom_mask[..., 1:3], dim=-1
    ) * torch.prod(all_atom_mask[..., :2], dim=-1)
    phi_mask = prev_all_atom_mask[..., 2] * torch.prod(
        all_atom_mask[..., :3], dim=-1, dtype=all_atom_mask.dtype
    )
    psi_mask = (
            torch.prod(all_atom_mask[..., :3], dim=-1, dtype=all_atom_mask.dtype)
            * all_atom_mask[..., 4]
    )

    chi_atom_indices = torch.as_tensor(
        get_chi_atom_indices(), device=aatype.device
    )

    atom_indices = chi_atom_indices[..., aatype, :, :]
    chis_atom_pos = batched_gather(
        all_atom_positions, atom_indices, -2, len(atom_indices.shape[:-2])
    )

    chi_angles_mask = all_atom_mask.new_tensor(rc.chi_angles_mask)

    chis_mask = chi_angles_mask[aatype, :]

    chi_angle_atoms_mask = batched_gather(
        all_atom_mask,
        atom_indices,
        dim=-1,
        no_batch_dims=len(atom_indices.shape[:-2]),
    )
    chi_angle_atoms_mask = torch.prod(
        chi_angle_atoms_mask, dim=-1, dtype=chi_angle_atoms_mask.dtype
    )
    chis_mask = chis_mask * chi_angle_atoms_mask

    torsions_atom_pos = torch.cat(
        [
            pre_omega_atom_pos[..., None, :, :],
            phi_atom_pos[..., None, :, :],
            psi_atom_pos[..., None, :, :],
            chis_atom_pos,
        ],
        dim=-3,
    )

    torsion_angles_mask = torch.cat(
        [
            pre_omega_mask[..., None],
            phi_mask[..., None],
            psi_mask[..., None],
            chis_mask,
        ],
        dim=-1,
    )

    torsion_frames = geometry.Rigid.from_3_points(
        torsions_atom_pos[..., 1, :],
        torsions_atom_pos[..., 2, :],
        torsions_atom_pos[..., 0, :],
        eps=1e-8,
    )

    # fourth_atom_rel_pos = torsion_frames.invert().apply(
    #    torsions_atom_pos[..., 3, :]
    # )
    fourth_atom_rel_pos = geometry.invert_rot_mul_vec(torsion_frames, torsions_atom_pos[..., 3, :])

    torsion_angles_sin_cos = torch.stack(
        [fourth_atom_rel_pos[..., 2], fourth_atom_rel_pos[..., 1]], dim=-1
    )

    denom = torch.sqrt(
        torch.sum(
            torch.square(torsion_angles_sin_cos),
            dim=-1,
            dtype=torsion_angles_sin_cos.dtype,
            keepdims=True,
        )
        + 1e-8
    )
    torsion_angles_sin_cos = torsion_angles_sin_cos / denom

    torsion_angles_sin_cos = torsion_angles_sin_cos * all_atom_mask.new_tensor(
        [1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
    )[((None,) * len(torsion_angles_sin_cos.shape[:-2])) + (slice(None), None)]

    chi_is_ambiguous = torsion_angles_sin_cos.new_tensor(
        rc.chi_pi_periodic,
    )[aatype, ...]

    mirror_torsion_angles = torch.cat(
        [
            all_atom_mask.new_ones(*aatype.shape, 3),
            1.0 - 2.0 * chi_is_ambiguous,
        ],
        dim=-1,
    )

    alt_torsion_angles_sin_cos = (
            torsion_angles_sin_cos * mirror_torsion_angles[..., None]
    )

    torsion_angles = torch.atan2(torsion_angles_sin_cos[..., 0], torsion_angles_sin_cos[..., 1])
    alt = torch.atan2(alt_torsion_angles_sin_cos[..., 0], alt_torsion_angles_sin_cos[..., 1])

    torsion_distance = chis_atom_pos[..., 1, :] - chis_atom_pos[..., 2, :]

    feat['torsion_angles'] = torsion_angles
    feat['torsion_angles_mask'] = torsion_angles_mask
    feat['torsion_distance'] = torsion_distance

    return feat
