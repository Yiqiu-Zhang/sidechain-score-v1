import io
import os
import pdbfixer
import multiprocessing
import threading
try:
    # openmm >= 7.6
    from openmm import app
    from openmm.app import element

except ImportError:
   # openmm < 7.6 (requires DeepMind patch)
   from simtk.openmm import app
   from simtk.openmm.app import element

def _remove_heterogens(fixer, alterations_info, keep_water):
    """Removes the residues that Pdbfixer considers to be heterogens.

    Args:
      fixer: A Pdbfixer instance.
      alterations_info: A dict that will store details of changes made.
      keep_water: If True, water (HOH) is not considered to be a heterogen.
    """
    initial_resnames = set()
    for chain in fixer.topology.chains():
        for residue in chain.residues():
            initial_resnames.add(residue.name)
    fixer.removeHeterogens(keepWater=keep_water)
    final_resnames = set()
    for chain in fixer.topology.chains():
        for residue in chain.residues():
            final_resnames.add(residue.name)
    alterations_info["removed_heterogens"] = initial_resnames.difference(
        final_resnames
    )


def fix_pdb(pdbfile, alterations_info, outfile):
    """Apply pdbfixer to the contents of a PDB file; return a PDB string result.

    1) Replaces nonstandard residues.
    2) Removes heterogens (non protein residues) including water.
    3) Adds missing residues and missing atoms within existing residues.
    4) Adds hydrogens assuming pH=7.0.
    5) KeepIds is currently true, so the fixer must keep the existing chain and
       residue identifiers. This will fail for some files in wider PDB that have
       invalid IDs.

    Args:
      pdbfile: Input PDB file handle.
      alterations_info: A dict that will store details of changes made.

    Returns:
      A PDB string representing the fixed structure.
    """
    fixer = pdbfixer.PDBFixer(filename= pdbfile)
    fixer.findNonstandardResidues()
    alterations_info["nonstandard_residues"] = fixer.nonstandardResidues
    fixer.replaceNonstandardResidues()
    _remove_heterogens(fixer, alterations_info, keep_water=False)
    fixer.findMissingResidues()
    alterations_info["missing_residues"] = fixer.missingResidues
    fixer.findMissingAtoms()
    alterations_info["missing_heavy_atoms"] = fixer.missingAtoms
    alterations_info["missing_terminals"] = fixer.missingTerminals
    fixer.addMissingAtoms(seed=0)
    return app.PDBFile.writeFile(
        fixer.topology, fixer.positions, open(outfile,'w'))

if __name__ =='__main__':
     
    
 #   directory1 =  './data/cath2/dompdb'
 #   directory2  = './data/temp1'
     
 #   for filename in os.listdir(directory1):
 #          file_path1 = os.path.join(directory1, filename)
 #          file_path1 = os.path.join(directory2, filename)
    
    path = './data/cath2/dompdb'
    path2 = './data/temp1'
    files = os.listdir(path)
    with open('/mnt/petrelfs/lvying/code/sidechain-diffusion/foldingdiff/wrong_file', 'r', encoding='utf-8') as f:
        wrong_pdb = f.readlines()
    i=0
    alterations_info = {}
    for filename in files:
        t = filename+'\n'
        if t in wrong_pdb:
          i = i + 1
          print(i)
          print(t)
          print(path+filename)
          fixed_pdb = fix_pdb(path+'/'+filename, alterations_info, path2+'/'+filename)
    
    
  #srun -p bio_s1 -n 1 --ntasks-per-node=1  --mem=256G --cpus-per-task=32  python  
    
        
    # fixed_pdb = fix_pdb('./data/cath/dompdb/2f4lA02', alterations_info, outfile)