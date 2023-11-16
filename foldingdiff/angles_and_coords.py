"""
Code to convert from angles between residues to XYZ coordinates. 
"""
import functools
import gzip
import os
import logging
import pickle
from typing import *
import warnings

import numpy as np
import pandas as pd

import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile

from foldingdiff import nerf

EXHAUSTIVE_ANGLES = ["phi", "psi", "omega", "tau", "CA:C:1N", "C:1N:1CA"]
EXHAUSTIVE_DISTS = ["0C:1N", "N:CA", "CA:C"]

MINIMAL_ANGLES = ["phi", "psi", "omega"]
MINIMAL_DISTS = []


def canonical_distances_and_dihedrals(
    fname: str,
    distances: List[str] = MINIMAL_DISTS,
    angles: List[str] = MINIMAL_ANGLES,
) -> Optional[pd.DataFrame]:
    """Parse the pdb file for the given values"""
    assert os.path.isfile(fname)
    warnings.filterwarnings("ignore", ".*elements were guessed from atom_.*")
    warnings.filterwarnings("ignore", ".*invalid value encountered in true_div.*")
    opener = gzip.open if fname.endswith(".gz") else open
    with opener(str(fname), "rt") as f:
        source = PDBFile.read(f)
    if source.get_model_count() > 1:
        return None
    # Pull out the atomarray from atomarraystack
    source_struct = source.get_structure()[0]

    # First get the dihedrals
    try:
        phi, psi, omega = struc.dihedral_backbone(source_struct)
        calc_angles = {"phi": phi, "psi": psi, "omega": omega}
    except struc.BadStructureError:
        logging.debug(f"{fname} contains a malformed structure - skipping")
        return None

    # Get any additional angles
    non_dihedral_angles = [a for a in angles if a not in calc_angles]
    # Gets the N - CA - C for each residue
    # https://www.biotite-python.org/apidoc/biotite.structure.filter_backbone.html
    backbone_atoms = source_struct[struc.filter_backbone(source_struct)]
    for a in non_dihedral_angles:
        if a == "tau" or a == "N:CA:C":
            # tau = N - CA - C internal angles
            idx = np.array(
                [list(range(i, i + 3)) for i in range(3, len(backbone_atoms), 3)]
                + [(0, 0, 0)]
            )
        elif a == "CA:C:1N":  # Same as C-N angle in nerf
            # This measures an angle between two residues. Due to the way we build
            # proteins out later, we do not need to meas
            idx = np.array(
                [(i + 1, i + 2, i + 3) for i in range(0, len(backbone_atoms) - 3, 3)]
                + [(0, 0, 0)]
            )
        elif a == "C:1N:1CA":
            idx = np.array(
                [(i + 2, i + 3, i + 4) for i in range(0, len(backbone_atoms) - 3, 3)]
                + [(0, 0, 0)]
            )
        else:
            raise ValueError(f"Unrecognized angle: {a}")
        calc_angles[a] = struc.index_angle(backbone_atoms, indices=idx)

    # At this point we've only looked at dihedral and angles; check value range
    for k, v in calc_angles.items():
        if not (np.nanmin(v) >= -np.pi and np.nanmax(v) <= np.pi):
            logging.warning(f"Illegal values for {k} in {fname} -- skipping")
            return None

    # Get any additional distances
    for d in distances:
        if (d == "0C:1N") or (d == "C:1N"):
            # Since this is measuring the distance between pairs of residues, there
            # is one fewer such measurement than the total number of residues like
            # for dihedrals. Therefore, we pad this with a null 0 value at the end.
            idx = np.array(
                [(i + 2, i + 3) for i in range(0, len(backbone_atoms) - 3, 3)]
                + [(0, 0)]
            )
        elif d == "N:CA":
            # We start resconstructing with a fixed initial residue so we do not need
            # to predict or record the initial distance. Additionally we pad with a
            # null value at the end
            idx = np.array(
                [(i, i + 1) for i in range(3, len(backbone_atoms), 3)] + [(0, 0)]
            )
            assert len(idx) == len(calc_angles["phi"])
        elif d == "CA:C":
            # We start reconstructing with a fixed initial residue so we do not need
            # to predict or record the initial distance. Additionally, we pad with a
            # null value at the end.
            idx = np.array(
                [(i + 1, i + 2) for i in range(3, len(backbone_atoms), 3)] + [(0, 0)]
            )
            assert len(idx) == len(calc_angles["phi"])
        else:
            raise ValueError(f"Unrecognized distance: {d}")
        calc_angles[d] = struc.index_distance(backbone_atoms, indices=idx)

    return pd.DataFrame({k: calc_angles[k].squeeze() for k in distances + angles})

@functools.lru_cache(maxsize=8192)
def get_pdb_length(fname: str) -> int:
    """
    Get the length of the chain described in the PDB file
    """
    warnings.filterwarnings("ignore", ".*elements were guessed from atom_.*")
    structure = PDBFile.read(fname)
    if structure.get_model_count() > 1:
        return -1
    chain = structure.get_structure()[0]
    backbone = chain[struc.filter_backbone(chain)]
    l = int(len(backbone) / 3)
    return l


def extract_backbone_coords(
    fname: str, atoms: Collection[Literal["N", "CA", "C"]] = ["CA"]
) -> Optional[np.ndarray]:
    """Extract the coordinates of the alpha carbons"""
    opener = gzip.open if fname.endswith(".gz") else open
    with opener(str(fname), "rt") as f:
        structure = PDBFile.read(f)
    if structure.get_model_count() > 1:
        return None
    chain = structure.get_structure()[0]
    backbone = chain[struc.filter_backbone(chain)]
    ca = [c for c in backbone if c.atom_name in atoms]
    coords = np.vstack([c.coord for c in ca])
    return coords


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # test_reverse_dihedral()
    print(
        extract_backbone_coords(
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/1CRN.pdb")
        )
    )
