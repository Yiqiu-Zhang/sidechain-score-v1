import pickle

from constant import restype_frame_mask, restype_rigid_group_default_frame
import geometry

import torch
import torch_geometric.data


full_data_name = 'cath_test.pkl'

with open(full_data_name, "rb") as file:

    proteins_list = pickle.load(file)

#Node feature: seq_emb, rigid_type, rigid_property
#Pair feature: distance_rbf, relative_pos
#Additional input: Rigid




def get_default_r(restype_idx):
    default_frame = torch.tensor(restype_rigid_group_default_frame)

    # [*, N, 8, 4, 4]
    res_default_frame = default_frame[restype_idx, ...]

    # [*, N, 8] Rigid
    default_r = geometry.from_tensor_4x4(res_default_frame)
    return default_r

def protein_to_graph(protein):
    angles = torch.as_tensor(protein['angles'])
    seq = torch.as_tensor(protein['seq'])
    coords = torch.as_tensor(protein['coords'])
    chi_mask = torch.as_tensor(protein['chi_mask'])
    rigid_type_onehot = torch.as_tensor(protein['rigid_type_onehot'])
    rigid_property = torch.as_tensor(protein['rigid_property'])
    acid_embedding = torch.as_tensor(protein['acid_embedding'])

    # rigid_mask
    restype_frame5_mask = torch.tensor(restype_frame_mask, dtype=bool)
    frame_mask = restype_frame5_mask[seq, ...]
    rigid_mask = torch.BoolTensor(torch.flatten(frame_mask, start_dim=-2))

    flat_rigid_type = rigid_type_onehot.reshape(-1, rigid_type_onehot.shape[-1])
    flat_rigid_property = rigid_property.reshape(-1, rigid_property.shape[-1])
    expand_seq = acid_embedding.repeat(1, 5).reshape(-1, acid_embedding.shape[-1])
    # [N_rigid, nf_dim] 6 + 20 + 320,
    node_feature = torch.cat((expand_seq, flat_rigid_type, flat_rigid_property), dim=-1).float()
    node_feature = node_feature[rigid_mask]

    res_index = torch.arange(0, len(seq))

    init_local_rigid = get_default_r(seq)

    data = torch_geometric.data.Data(x = node_feature,
                                     true_chi=angles,
                                     aatype=seq,
                                     bb_coord=coords,
                                     local_rigid = init_local_rigid,
                                     chi_mask = chi_mask,
                                     rigid_mask = rigid_mask,
                                     res_index = res_index,
                                     )

    return data







