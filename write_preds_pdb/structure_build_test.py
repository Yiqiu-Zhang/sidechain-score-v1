
import torch
import sys
sys.path.append(r"/mnt/petrelfs/zhangyiqiu/sidechain-score-v1")

from model import dataset
from torch_geometric.transforms import BaseTransform
from structure_build_score import write_pdb_from_position, frame_to_pos

class EmptyTransform(BaseTransform):
    def __init__(self):

        self.ramdom_sample = torch.distributions.uniform.Uniform(-torch.pi, torch.pi)

    def __call__(self, protein):
        
        protein, _ = dataset.transform_structure(protein, protein.true_chi)

        return protein

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(sigma_min={self.sigma_min}, '
                f'sigma_max={self.sigma_max})')
    

test_graph_name = '/mnt/petrelfs/zhangyiqiu/sidechain-score-v1/foldingdiff/test_graph.pkl'
graph_data = dataset.preprocess_datapoints(graph_data = test_graph_name)

real_transform = EmptyTransform()
data = dataset.ProteinDataset(data = graph_data, transform=real_transform)
protein = data[0]
zero = torch.ones(protein.true_chi.shape)
protein, all_frames_to_global = dataset.transform_structure(protein, 0.5*zero)

all_atom_positions = frame_to_pos(all_frames_to_global, 
                                                  protein.aatype,
                                                  protein.bb_coord)

out_dir = '/mnt/petrelfs/zhangyiqiu/sidechain-score-v1/bin'
from pathlib import Path
pdbname = Path(protein.fname).name
write_pdb_from_position(protein, all_atom_positions, out_dir, pdbname, 0)