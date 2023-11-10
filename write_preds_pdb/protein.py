import dataclasses
from typing import Sequence, Optional
import string


import write_preds_pdb.constant as constant
import numpy as np
import io
from Bio.PDB import PDBParser

@dataclasses.dataclass(frozen=True)
class Protein:
    """Protein structure representation."""

    # Cartesian coordinates of atoms in angstroms. The atom types correspond to
    # constant.atom_types, i.e. the first three are N, CA, CB.
    atom_positions: np.ndarray  # [num_res, num_atom_type, 3]

    # Amino-acid type for each residue represented as an integer between 0 and
    # 20, where 20 is 'X'.
    aatype: np.ndarray  # [num_res]

    # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
    # is present and 0.0 if not. This should be used for loss masking.
    atom_mask: np.ndarray  # [num_res, num_atom_type]

    # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
    residue_index: np.ndarray  # [num_res]

    # B-factors, or temperature factors, of each residue (in sq. angstroms units),
    # representing the displacement of the residue from its ground truth mean
    # value.
    b_factors: np.ndarray  # [num_res, num_atom_type]

    # Chain indices for multi-chain predictions
    chain_index: Optional[np.ndarray] = None

    # Optional remark about the protein. Included as a comment in output PDB 
    # files
    remark: Optional[str] = None

    # Templates used to generate this protein (prediction-only)
    parents: Optional[Sequence[str]] = None

    # Chain corresponding to each parent
    parents_chain_index: Optional[Sequence[int]] = None

def to_pdb(prot: Protein) -> str:
    """Converts a `Protein` instance to a PDB string.

    Args:
      prot: The protein to convert to PDB.

    Returns:
      PDB string.
    """
   # restypes = constant.restypes + ["X"]
    restypes = constant.restypes
    res_1to3 = lambda r: constant.restype_1to3.get(restypes[r], "UNK")
    atom_types = constant.atom_types

    pdb_lines = []

    atom_mask = prot.atom_mask
    aatype = prot.aatype
    atom_positions = prot.atom_positions
    residue_index = prot.residue_index.astype(np.int32)
    b_factors = prot.b_factors
    chain_index = prot.chain_index

    if np.any(aatype > constant.restype_num):
        raise ValueError("Invalid aatypes.")

    n = aatype.shape[0]
    atom_index = 1
    prev_chain_index = 0
    chain_tags = string.ascii_uppercase
    # Add all atom sites.
    for i in range(n):
        res_name_3 = res_1to3(aatype[i])
        for atom_name, pos, mask in zip(
                atom_types, atom_positions[i], atom_mask[i] # 拿掉b_factor
        ):
            if mask < 0.5:  # 这个是用来干什么的 atom 被 mask掉就跳过
                continue

            record_type = "ATOM"
            name = atom_name if len(atom_name) == 4 else f" {atom_name}"
            alt_loc = ""
            insertion_code = ""
            occupancy = 1.00
            b_factor = 100.00 # 这个b_factor是写死的
            element = atom_name[
                0
            ]  # Protein supports only C, N, O, S, this works.
            charge = ""

            chain_tag = "A"

            if chain_index is not None:
                chain_tag = chain_tags[chain_index[i]]

            # PDB is a columnar format, every space matters here!
            atom_line = (
                f"{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}"
                f"{res_name_3:>3} {chain_tag:>1}"
                f"{residue_index[i]:>4}{insertion_code:>1}   "
                f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                f"{element:>2}{charge:>2}"
            )
            pdb_lines.append(atom_line)
            atom_index += 1

        should_terminate = (i == n - 1)
        if chain_index is not None:
            if i != n - 1 and chain_index[i + 1] != prev_chain_index:
                should_terminate = True
                prev_chain_index = chain_index[i + 1]

        if should_terminate:
            # Close the chain.
            chain_end = "TER"
            chain_termination_line = (
                f"{chain_end:<6}{atom_index:>5}      "
                f"{res_1to3(aatype[i]):>3} "
                f"{chain_tag:>1}{residue_index[i]:>4}"
            )
            pdb_lines.append(chain_termination_line)
            atom_index += 1

    pdb_lines.append("END")
    pdb_lines.append("")
    return "\n".join(pdb_lines)

def from_pdb_string(pdb_path: str, chain_id: Optional[str] = None) -> Protein:
    """Takes a PDB string and constructs a Protein object.

    WARNING: All non-standard residue types will be converted into UNK. All
      non-standard atoms will be ignored.

    Args:
      pdb_str: The contents of the pdb file
      chain_id: If None, then the pdb file must contain a single chain (which
        will be parsed). If chain_id is specified (e.g. A), then only that chain
        is parsed.

    Returns:
      A new `Protein` parsed from the pdb contents.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("name", pdb_path)
    model = structure[0]
    chain = model.child_list[0]


    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    chain_ids = []
    b_factors = []

    removelist = []
    for res in chain:
        if res.id[0] != " ": 
            removelist.append(res.id)
        elif res.id[2] != " ": # avoid insertion code
            removelist.append(res.id)
    for id in removelist:
        chain.detach_child(id)

    for res in chain:

        res_shortname = constant.restype_3to1.get(res.resname, "X")
        restype_idx = constant.restype_order.get(
            res_shortname, constant.restype_num
        )
        pos = np.zeros((constant.atom_type_num, 3))
        mask = np.zeros((constant.atom_type_num,))
        res_b_factors = np.zeros((constant.atom_type_num,))
        for atom in res:
            if atom.name not in constant.atom_types:
                continue
            pos[constant.atom_order[atom.name]] = atom.coord
            mask[constant.atom_order[atom.name]] = 1.0
            res_b_factors[
                constant.atom_order[atom.name]
            ] = atom.bfactor
        if np.sum(mask) < 0.5:
            # If no known atom positions are reported for the residue then skip it.
            continue
        aatype.append(restype_idx)
        atom_positions.append(pos)
        atom_mask.append(mask)
        residue_index.append(res.id[1])
        chain_ids.append(chain.id)
        b_factors.append(res_b_factors)


    return Protein(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        residue_index=np.array(residue_index),
        b_factors=np.array(b_factors),
    )

