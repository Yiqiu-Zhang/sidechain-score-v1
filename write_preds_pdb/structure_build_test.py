from structure_build import torsion_to_position, torsion_to_frame, frame_to_edge

from foldingdiff import PDB_processing
#from foldingdiff.PDB_processing import get_torsion_seq
import PDB_processing_with_unk
import numpy as np
import torch
from constant import restype_atom37_mask
import protein
import os
import torch.nn.functional as F


data_path1 = '/mnt/petrelfs/lvying/code/sidechain-rigid-attention_yiqiu/data/standard_results/native/casp_all/'
data_path2 = '/mnt/petrelfs/lvying/code/sidechain-rigid-attention_yiqiu/data/standard_results/our_method/ipa_noise_generate_6.14/'
path1 = 'T0735-D2'
path2 = f'{data_path2}reconstructed_{path1}.pdb'

#features = PDB_processing_with_unk.get_torsion_seq(data_path+path1)
features = PDB_processing_with_unk.get_torsion_seq(data_path1 + path1+'.pdb')
angles = torch.tensor(features["angles"].to_numpy())
bb_pos = torch.tensor(features["coords"])
seq = torch.tensor(features["seq"])

angles = F.pad(input=angles,pad=(0,0,0,4),value=0)
bb_pos = F.pad(input=bb_pos,pad=(0,0,0,0,0,4),value=0)
seq = F.pad(input=seq,pad=(0,4),value=0)

final_atom_positions = torsion_to_position(seq,
                                           bb_pos,
                                           angles)
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

'''
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