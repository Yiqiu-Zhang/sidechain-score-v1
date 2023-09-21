from rigid_diffusion_score import RigidDiffusion
import torch
from write_preds_pdb import  structure_build_score as structure_build, geometry
batch = 5
n_rigid = 100
n_res = 20
side_chain_angles = torch.randn((batch,n_res,4))
backbone_coords = torch.randn((batch,n_res,4,3))
seq_idx = torch.randint(0,20,(batch,n_res))
seq_esm = torch.randn((batch,n_res,320))
sigma = torch.rand((batch))* torch.pi
#timesteps = torch.randint(0,1000,(batch,1))
rigid_type = torch.randn((batch, n_res, 5, 19))
rigid_property = torch.randn((batch,n_res,5,7))
pad_mask = torch.ones(batch,n_res)
ture_angles = torch.randn((batch,n_res,4))
diffusion_mask = torch.randint(0,1, (batch,n_res,1)) == 1

# [*, N_rigid] Rigid
rigids, current_local_frame, all_frames_to_global = structure_build.torsion_to_frame(side_chain_angles,
                                                                                     seq_idx,
                                                                                     backbone_coords)
model = RigidDiffusion()
node_emb, sum_local_trans = model.forward(rigids,
                    seq_idx,
                    sigma,
                    seq_esm,
                    rigid_type,
                    rigid_property,
                    pad_mask)

