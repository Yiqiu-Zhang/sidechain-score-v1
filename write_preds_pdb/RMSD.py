import Bio.PDB

def _convert_mse(structure):
    # Changes MSE residues to MET
    for residue in Bio.PDB.Selection.unfold_entities(structure, 'R'):
        if residue.get_resname() == 'MSE':
            residue.resname = 'MET'
            for atom in residue:
                if atom.element == 'SE':
                    new_atom = Bio.PDB.Atom.Atom('SD',
                                         atom.coord,
                                         atom.bfactor,
                                         atom.occupancy,
                                         atom.altloc,
                                         'SD  ',
                                         atom.serial_number,
                                         element='S')
                    residue.add(new_atom)
                    atom_to_remove = atom.get_id()
                    residue.detach_child(atom_to_remove)
"""	
    def RMSD(structure_1, structure_2):
        '''
        Calculate the RMSD between two protein structures using Biopython
        The Biopython algorithm is poorly designed and only aligns local motifs
        rather than full protein structures/complexes.
        '''
    
        builder = Bio.PDB.Polypeptide.CaPPBuilder()
        STR1 = builder.build_peptides(Bio.PDB.PDBParser(QUIET=True)\
            .get_structure('Structure 1', structure_1), aa_only=True)
        STR2 = builder.build_peptides(Bio.PDB.PDBParser(QUIET=True)\
            .get_structure('Structure 2', structure_2), aa_only=True)
        fixed  = [atom for poly in STR1 for res in poly for atom in res.sort()]
        moving = [atom for poly in STR2 for res in poly for atom in res.sort()]
        lengths = [len(fixed), len(moving)]
        smallest = min(lengths)
        sup = Bio.PDB.Superimposer()
        sup.set_atoms(fixed[:smallest], moving[:smallest])
        sup.apply(Bio.PDB.PDBParser(QUIET=True)\
            .get_structure('Structure 2', structure_2)[0].get_atoms())
        RMSD = round(sup.rms, 4)
        print(RMSD)
"""

def RMSD_single_chain(path_1, path_2):

    parser = Bio.PDB.PDBParser(QUIET=True)
    structure1 = parser.get_structure('name', path_1)
    _convert_mse(structure1)
    model1 = structure1[0]
    chain1 = model1.child_list[0]
    structure2 = parser.get_structure('name', path_2)
    _convert_mse(structure2)
    model2 = structure2[0]
    chain2 = model2.child_list[0]

    fixed = [atom  for res in chain1 for atom in sorted(res)]
    moving = [atom for res in chain2 for atom in sorted(res)]

    lengths = [len(fixed), len(moving)]
    smallest = min(lengths)
    sup = Bio.PDB.Superimposer()
    sup.set_atoms(fixed[:smallest], moving[:smallest])
    sup.apply(parser.get_structure('Structure 2', path_2)[0].get_atoms())
    RMSD = round(sup.rms, 4)
    print(RMSD)


name = '17gsA01'
path1 = f'/home/PJLAB/zhangyiqiu/PycharmProjects/sidechain-rigid-attention/write_preds_pdb/test/reconstructed_{name}.pdb'
print(name)
print()
for i in range(10):
    path2 = f'/home/PJLAB/zhangyiqiu/PycharmProjects/sidechain-rigid-attention/write_preds_pdb/test/IPA_pdb_6.13/{name}_generate_{i}.pdb'
    path3  = f'/home/PJLAB/zhangyiqiu/PycharmProjects/sidechain-rigid-attention/write_preds_pdb/test/sampled_pdb_5_16/{name}_generate_{i}.pdb'
    RMSD_single_chain(path2,path1)
    RMSD_single_chain(path3, path1)
    print('5_16 ############################################')