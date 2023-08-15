from structure_build_score import torsion_to_position, torsion_to_frame, frame_to_edge
import structure_build_score as structure_build
#from foldingdiff import PDB_processing
#from foldingdiff.PDB_processing import get_torsion_seq
import PDB_processing_with_unk
import numpy as np
import torch
from constant import restype_atom37_mask
import protein
import os
import torch.nn.functional as F
import geometry
import sys



data_path1 = '/mnt/petrelfs/lvying/code/sidechain-rigid-attention_yiqiu/data/standard_results/native/casp_all/'
data_path2 = '/mnt/petrelfs/lvying/code/sidechain-rigid-attention_yiqiu/data/standard_results/our_method/ipa_noise_generate_6.14/'
path1 = 'T0735-D2'
path2 = f'{data_path2}reconstructed_{path1}.pdb'

#features = PDB_processing_with_unk.get_torsion_seq(data_path+path1)
features = PDB_processing_with_unk.get_torsion_seq('1a02F00')
angles = torch.tensor(features["angles"].to_numpy()).unsqueeze(0).repeat(2, 1, 1)
bb_pos = torch.tensor(features["coords"]).unsqueeze(0).repeat(2,1,1,1)
seq = torch.tensor(features["seq"]).unsqueeze(0).repeat(2,1)
chi_mask = torch.tensor((features["chi_mask"])).unsqueeze(0).repeat(2, 1, 1)


#angles = F.pad(input=angles,pad=(0,0,0,4),value=0)
#bb_pos = F.pad(input=bb_pos,pad=(0,0,0,0,0,4),value=0)
#seq = F.pad(input=seq,pad=(0,4),value=0)
default_r = structure_build.get_default_r(seq, angles)
'''
final_atom_positions = torsion_to_position(seq,
                                           bb_pos,
                                           angles,
                                           default_r)
'''
bb_to_gb = geometry.get_gb_trans(bb_pos)
angles_sin_cos = torch.stack(
    [
        torch.sin(angles),
        torch.cos(angles),
    ],
    dim=-1,
)

rigid, current_local_frame = structure_build.rigid_apply_update(seq, bb_to_gb, angles_sin_cos, default_r)

angles_from_rigid = structure_build.rigids_to_torsion_angles(seq, rigid)
print('angles_from_rigid',angles_from_rigid.shape)

'''
chain_len = len(seq)
index = np.arange(1,chain_len+1)
features["final_atom_mask"] = restype_atom37_mask[features["seq"]]
resulted_protein = protein.Protein(
    aatype= features["seq"],  # [*,N]
    atom_positions=final_atom_positions,
    atom_mask=features["final_atom_mask"],
    residue_index=index,  # [num_res]
    b_factors=np.zeros_like(features["final_atom_mask"]))

pdb_str = protein.to_pdb(resulted_protein)

with open(path2, 'w') as fp:
    fp.write(pdb_str)


angles = torch.tensor(features["angles"].to_numpy()).unsqueeze(0)
bb_pos = torch.tensor(features["coords"]).unsqueeze(0)
seq = torch.tensor(features["seq"]).unsqueeze(0)
angles = F.pad(input=angles,pad=(0,0,0,4),value=0)
bb_pos = F.pad(input=bb_pos,pad=(0,0,0,0,0,4),value=0)
seq = F.pad(input=seq,pad=(0,4),value=0)

rigid_by_residue = torsion_to_frame(seq, bb_pos, angles)

frame_pair_mask, distance, altered_direction, orientation = frame_to_edge(
    rigid_by_residue, seq)

RMSD.RMSD_single_chain(path2,data_path+ path1)'''

atom_types = [
    "N",
    "CA",
    "C",
    "CB",
    "O",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
]
'''
atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}

from Bio import PDB

def get_atom_coords(
    pdb_path,
    chain_id = "F",
    _zero_center_positions: bool = False
):
    # Locate the right chain
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('name', pdb_path)
    model = structure[0]
    chain = model.child_list[0]

    # Extract the coordinates
    num_res = len(chain)
    all_atom_positions = np.zeros(
        [num_res, 37, 3], dtype=np.float32
    )
    all_atom_mask = np.zeros(
        [num_res, 37], dtype=np.float32
    )
    for res_index in range(num_res):
        pos = np.zeros([37, 3], dtype=np.float32)
        mask = np.zeros([37], dtype=np.float32)
        res = chain[res_index+140]

        for atom in res.get_atoms():
            atom_name = atom.get_name()
            x, y, z = atom.get_coord()
            if atom_name in atom_order.keys():
                pos[atom_order[atom_name]] = [x, y, z]
                mask[atom_order[atom_name]] = 1.0
            elif atom_name.upper() == "SE" and res.get_resname() == "MSE":
                # Put the coords of the selenium atom in the sulphur column
                pos[atom_order["SD"]] = [x, y, z]
                mask[atom_order["SD"]] = 1.0

        all_atom_positions[res_index] = pos
        all_atom_mask[res_index] = mask

    if _zero_center_positions:
        binary_mask = all_atom_mask.astype(bool)
        translation_vec = all_atom_positions[binary_mask].mean(axis=0)
        all_atom_positions[binary_mask] -= translation_vec

    return all_atom_positions, all_atom_mask

atom_pos, atom_mask = get_atom_coords('1a02F00')

angle_openfold, alt = structure_build.atom37_to_torsion_angles(seq,atom_pos,atom_mask)

openfold_atom_positions = torsion_to_position(seq,
                                           bb_pos,
                                           angle_openfold[:,3:],
                                           default_r)

diff_open = torch.sum(torch.sqrt((torch.tensor(atom_pos) - openfold_atom_positions)**2),dim=(-1,-2,-3))
diff_self = torch.sum(torch.sqrt((torch.tensor(atom_pos) - final_atom_positions)**2),dim=(-1,-2,-3))
'''