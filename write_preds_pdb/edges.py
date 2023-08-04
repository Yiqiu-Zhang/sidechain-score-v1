'''
This file is used to calculate the frame dependency of the attention module
'''

import pickle
import structure_build
import torch
import numpy as np

with open('esm3B_cache_canonical_structures_cath_2190349f44fee70c6e1eda000be2bdd4.pkl', 'rb') as f:
    data = pickle.load(f)[1]

aatype_idx = np.array([k['seq'] for k in data])
bb_coords =np.array([k['coords'] for k in data])
side_chain_angles = np.array([k['angles'].values for k in data])

seq_pad = [seq[:30] for seq in aatype_idx]
bb_coords_pad = [seq[:30] for seq in bb_coords]
side_chain_angles_pad = [seq[:30] for seq in side_chain_angles]


side_chain_angles = torch.tensor(side_chain_angles_pad)
bb_coords = torch.tensor(bb_coords_pad)
aatype_idx = torch.tensor(seq_pad)



rigid_by_residue = structure_build.torsion_to_frame(aatype_idx, bb_coords, side_chain_angles) # add attention

frame_pair_mask, distance, altered_direction, orientation = structure_build.frame_to_edge(rigid_by_residue, aatype_idx) #rigid feature


