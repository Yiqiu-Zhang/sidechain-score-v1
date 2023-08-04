from model import RigidDiffusion
import torch
batch = 2
n_rigid = 25
n_res = 5
side_chain_angles = torch.randn((batch,n_res,4))
backbone_coords = torch.randn((batch,n_res,4,3))
seq_idx = torch.randint(0,20,(batch,n_res))
seq_esm = torch.randn((batch,n_res,320))

timesteps = torch.randint(0,1000,(batch,1))
rigid_type = torch.randn((batch, n_res, 5, 20))
rigid_property = torch.randn((batch,n_res,5,6))
pad_mask = torch.randint(0,1, (batch,n_res))
model = RigidDiffusion()
run = model.forward(side_chain_angles,
                    backbone_coords,
                    seq_idx,
                    timesteps,
                    seq_esm,
                    rigid_type,
                    rigid_property,
                    pad_mask,)

