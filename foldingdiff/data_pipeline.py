
from write_preds_pdb import constant
import numpy as np
from write_preds_pdb import protein
from typing import Mapping, Optional
import torch
import torch.nn.functional as F
from write_preds_pdb import structure_build_score as sb
FeatureDict = Mapping[str, np.ndarray]

def make_rigid_feature(sequence):
    features = {}
    res_name = [constant.restype_1to3[i] for i in sequence]
    L = len(res_name)
    rigid_type = np.zeros((L,5))
    rigid_type_mask = np.zeros((L,5))
    rigid_property = np.zeros((L,5,6))

    for res_idx, name in enumerate(res_name):
        res_rigid_group_list = constant.restype_name_to_rigid_idx[name]

        for i, rigid in enumerate(res_rigid_group_list):
            rigid_type[res_idx][i] = rigid
            rigid_type_mask[res_idx][i] = 1
            rigid_property[res_idx][i][:] = constant.rigid_type_property[rigid]

    rigid_type = torch.tensor(rigid_type, dtype=torch.int64)
    rigid_type_onehot = F.one_hot(rigid_type, 21)  # with the empty rigid type 0
    rigid_type_onehot = rigid_type_onehot * torch.unsqueeze(torch.tensor(rigid_type_mask), -1)

    rigid_type_onehot = np.array(rigid_type_onehot)  # (L,5,20)
    features['rigid_type_onehot'] = rigid_type_onehot
    features['rigid_property'] = rigid_property

    return features

def make_protein_features(
    protein_object: protein.Protein,
    _is_distillation: bool = False,
) -> FeatureDict:
    pdb_feats = {}
    aatype = protein_object.aatype
    sequence = [constant.restypes[i]for i in aatype]

    pdb_feats["seq"] = aatype
    pdb_feats["seq_temp"] = np.array(sequence)
    pdb_feats['residue_index'] = protein_object.residue_index

    all_atom_positions = protein_object.atom_positions
    all_atom_mask = protein_object.atom_mask

    pdb_feats["all_atom_positions"] = all_atom_positions.astype(np.float32)
    if len(all_atom_positions.shape) != 3:
         raise ValueError(f'{all_atom_positions.shape}'f' positions {pdb_feats["all_atom_positions"]}')
    pdb_feats["coords"] = pdb_feats['all_atom_positions'][:,[0,1,2,4],:]
    pdb_feats["all_atom_mask"] = all_atom_mask.astype(np.float32)

    pdb_feats["resolution"] = np.array([0.]).astype(np.float32)
    pdb_feats["is_distillation"] = np.array(
        1. if _is_distillation else 0.
    ).astype(np.float32)

    pdb_feats.update(
        sb.atom37_to_torsion_feature(torch.tensor(aatype),
                                    all_atom_positions,
                                    all_atom_mask)
    )
    pdb_feats["angles"] = pdb_feats["torsion_angles"][...,3:]
    pdb_feats['chi_mask'] = pdb_feats["torsion_angles_mask"][...,3:]
    pdb_feats.update(make_rigid_feature(sequence))

    return pdb_feats

def make_pdb_features(
    protein_object: protein.Protein,
    is_distillation: bool = True,
    confidence_threshold: float = 50.,
) -> FeatureDict:
    pdb_feats = make_protein_features(
        protein_object, _is_distillation=True
    )

    if is_distillation:
        high_confidence = protein_object.b_factors > confidence_threshold
        high_confidence = np.any(high_confidence, axis=-1)
        pdb_feats["all_atom_mask"] *= high_confidence[..., None]

    return pdb_feats

def process_pdb(
        pdb_path: str,
        is_distillation: bool = True,
        chain_id: Optional[str] = None,
) -> FeatureDict:
    """
        Assembles features for a protein in a PDB file.
    """

    with open(pdb_path, 'r') as f:
        pdb_str = f.read()

    protein_object = protein.from_pdb_string(pdb_str, chain_id)
    pdb_feats = make_pdb_features(
        protein_object,
        is_distillation=is_distillation
    )

    return {**pdb_feats}

'''
from structure_build_score import torsion_to_position, get_default_r, write_pdb_from_position
test = '1a02F00.pdb'
recon = '/home/PJLAB/zhangyiqiu/PycharmProjects/sidechain-score/data/reconstructed/reconstructed_1a02F00.pdb'
from os import listdir
from os.path import isfile, join
mypath = '/home/PJLAB/zhangyiqiu/PycharmProjects/sidechain-score/data/native/casp_fixed'
fnames = [f for f in listdir(mypath) if isfile(join(mypath, f))]

import pickle
with open('/home/PJLAB/zhangyiqiu/PycharmProjects/openfold/tests/openfold.pkl', 'rb') as f:

    of= pickle.load(f)

for fname in fnames:
    location = mypath +'/'+ fname

    a = process_pdb(location)
    default_r = get_default_r(a['seq'], a['angles'])
    test_angle = torch.zeros(53,4)
    b = torsion_to_position(torch.tensor(a['seq']),
                            torch.tensor(a['coords']),
                            a['angles'],
                            #test_angle,
                            default_r)

    write_pdb_from_position(a, b,
                            '/home/PJLAB/zhangyiqiu/PycharmProjects/sidechain-score/data/reconstructed',
                            'reconstructed_'+fname,
                            0)


a_test = process_pdb('/home/PJLAB/zhangyiqiu/PycharmProjects/sidechain-score/write_preds_pdb/test_no_real_bb/reconstructed_1a02F00.pdb')

default_r = get_default_r(a_test['seq'], a_test['angles'])
all_frames_to_global,all_pos,b = torsion_to_position(torch.tensor(a_test['seq']),
                        torch.tensor(a_test['coords']),
                        a_test['angles'],
                        of['pred_xyz'][0],
                        default_r)

write_pdb_from_position(a_test, b,
                        '/home/PJLAB/zhangyiqiu/PycharmProjects/sidechain-score/data/reconstructed',
                        'reconstructed_re_'+test_real_bb,
                        0)



test_real_bb = of['pred_xyz'][0] - all_pos
pos_test = torch.sum(abs(test_real_bb),dim =-1)
test_frame = of['all_frames_to_global'][0] - all_frames_to_global.rot.get_rot_mat()

frame = torch.sum(abs(test_frame),dim =(-1,-2))


# region Description
import pickle
a_torsion = []
with open('1a_torsionangle', 'r') as f:

    lines = f.readlines()[1:]
for line in lines:
    temp = line.strip().split(',')[5:]

    a_torsion.append([float(i)  if i != '' else 0. for i in temp])

angles = a['angles']
y = angles[angles[:, 0] != 0.]

C = y - np.array(a_torsion)

# endregion

for i in range(37):
    print(torch.max(torch.sqrt(
        torch.sum(torch.tensor(a['all_atom_positions'][:, i] - a_test['all_atom_positions'][:, i]) ** 2, dim=-1))))
'''