import torch
from Bio import PDB
import math
import numpy as np
import pandas as pd
import torch.nn.functional as F
from write_preds_pdb import constant as rc


def acid_to_number(seq, mapping):
    num_list = []
    for acid in seq:
        if acid in mapping:
            num_list.append(mapping[acid])
    return num_list

def rigid_distance_pdb(pdb_path):

    torsion_distance_list = []
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('name', pdb_path)
    model = structure[0]
    chain = model.child_list[0]
    removelist = []

    for res in chain:
        if res.id[0] != " ":
            removelist.append(res.id)
    for id in removelist:
        chain.detach_child(id)

    for res_idx, res in enumerate(chain):
        dihedral_vec_list = [[0,0,0]] * 4


        res_name = res.resname
        res_torsion_atom_list = rc.chi_angles_atoms[res_name]

        for i, torsion_atoms in enumerate(res_torsion_atom_list):
            # 这个还有点问题，因为在这个坐标应该是用上一个rigid的local frame来定义的，
            # 这里直接用了 global的坐标去减。 如果单纯考虑 r的大小的话不重要，
            # 但是要是之后要考虑去做成vector loss的话就有点问题了
            vec_atoms_coord = [res[a].get_vector() for a in torsion_atoms]
            dihedral_vec =  vec_atoms_coord[1] - vec_atoms_coord[2]
            dihedral_vec_list[i] = dihedral_vec.get_array()
        torsion_distance_list.append(dihedral_vec_list)

    torsion_distance = np.array(torsion_distance_list)

    return torsion_distance

'''
def get_torsion_seq(pdb_path):
    
    torsion_list = []
    chi_mask =[]
    seq = []
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('name', pdb_path)
    model = structure[0]
    chain = model.child_list[0]
    X = []
    removelist = []
    for res in chain:
        if res.id[0] != " ":
            removelist.append(res.id)
    for id in removelist:
        chain.detach_child(id)
    L = len(chain)
    rigid_type = np.zeros((L,5))
    rigid_type_mask = np.zeros((L,5))
    rigid_property = np.zeros((L,5,6))

    for res_idx, res in enumerate(chain):

        chi_list = [0] * 4
        temp_mask =[0] * 4
        

        res_name = res.resname
        seq.append(rc.restype_3to1[res_name])
        res_torsion_atom_list = chi_angles_atoms[res_name]
        res_rigid_group_list = rc.restype_name_to_rigid_idx[res_name]
        X.append([res[a].get_coord() for a in rc.bb_atoms])

        for i, rigid in enumerate(res_rigid_group_list):
            rigid_type[res_idx][i] = rigid
            rigid_type_mask[res_idx][i] = 1
            rigid_property[res_idx][i][:] = rigid_type_property[rigid]

        for i, torsion_atoms in enumerate(res_torsion_atom_list):
            vec_atoms_coord = [res[a].get_vector() for a in torsion_atoms]
            angle = PDB.calc_dihedral(*vec_atoms_coord)
            chi_list[i] = angle
            temp_mask[i] = 1
        torsion_list.append(chi_list)
        chi_mask.append(temp_mask)
    chi_mask = np.array(chi_mask)
    torsion_list = np.array(torsion_list)
    X = np.array(X)
    
    #=========
    seq_single = np.array(seq, dtype=np.str)
    
    num_acid_seq = acid_to_number(seq_single, AA_TO_ID)
    num_acid_seq = np.array(num_acid_seq)
    
    X1 = torsion_list[:,0]
    X2 = torsion_list[:,1]
    X3 = torsion_list[:,2]
    X4 = torsion_list[:,3]
    calc_angles = {"X1": X1, "X2": X2, "X3": X3, "X4": X4}
    angle_list = pd.DataFrame({k: calc_angles[k].squeeze() for k in ANGLES})

    rigid_type = torch.tensor(rigid_type, dtype=torch.int64)
    rigid_type_onehot = F.one_hot(rigid_type,20)  # with the empty rigid type 0 
    rigid_type_onehot = rigid_type_onehot * torch.unsqueeze(torch.tensor(rigid_type_mask), -1)

    rigid_type_onehot = np.array(rigid_type_onehot) #(L,5,20)

    dict_struct = {'angles': angle_list,
                   'coords': X,
                   'seq': num_acid_seq,
                   "seq_temp": seq_single,
                   "chi_mask": chi_mask, # [L,4]
                   'rigid_type_onehot': rigid_type_onehot, #(L,5,20)
                   'rigid_property': rigid_property, # (L,5,6)
                   'fname': pdb_path,
                   }
    return dict_struct
'''

#t = get_torsion_seq('./data/1CRN.pdb')
#l = len(t1)
#l2 = len(t['seq'])
#seq= "".join(t["seq"])
#print(seq)  chi_1