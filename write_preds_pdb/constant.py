import numpy as np
import torch

restypes = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P',
            'S', 'T', 'W', 'Y', 'V', 'X'] # with the UNK residue

restype_order = {restype: i for i, restype in enumerate(restypes)}


restype_num = len(restypes) # 20 # Now 21 with UNK res

restype_1to3 = {
    'A': 'ALA',
    'R': 'ARG',
    'N': 'ASN',
    'D': 'ASP',
    'C': 'CYS',
    'Q': 'GLN',
    'E': 'GLU',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'L': 'LEU',
    'K': 'LYS',
    'M': 'MET',
    'F': 'PHE',
    'P': 'PRO',
    'S': 'SER',
    'T': 'THR',
    'W': 'TRP',
    'Y': 'TYR',
    'V': 'VAL',
    'X': 'UNK', #With UNK res as a restype
}

# A compact atom encoding with 14 columns
# pylint: disable=line-too-long
# pylint: disable=bad-whitespace
restype_name_to_atom14_names = {
    'ALA': ['N', 'CA', 'C', 'O', 'CB', '',    '',    '',    '',    '',    '',    '',    '',    ''],
    'ARG': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'NE',  'CZ',  'NH1', 'NH2', '',    '',    ''],
    'ASN': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'ND2', '',    '',    '',    '',    '',    ''],
    'ASP': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'OD2', '',    '',    '',    '',    '',    ''],
    'CYS': ['N', 'CA', 'C', 'O', 'CB', 'SG',  '',    '',    '',    '',    '',    '',    '',    ''],
    'GLN': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'NE2', '',    '',    '',    '',    ''],
    'GLU': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'OE2', '',    '',    '',    '',    ''],
    'GLY': ['N', 'CA', 'C', 'O', '',   '',    '',    '',    '',    '',    '',    '',    '',    ''],
    'HIS': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'ND1', 'CD2', 'CE1', 'NE2', '',    '',    '',    ''],
    'ILE': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', '',    '',    '',    '',    '',    ''],
    'LEU': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', '',    '',    '',    '',    '',    ''],
    'LYS': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'CE',  'NZ',  '',    '',    '',    '',    ''],
    'MET': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'SD',  'CE',  '',    '',    '',    '',    '',    ''],
    'PHE': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  '',    '',    ''],
    'PRO': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  '',    '',    '',    '',    '',    '',    ''],
    'SER': ['N', 'CA', 'C', 'O', 'CB', 'OG',  '',    '',    '',    '',    '',    '',    '',    ''],
    'THR': ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', '',    '',    '',    '',    '',    '',    ''],
    'TRP': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
    'TYR': ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  'OH',  '',    ''],
    'VAL': ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', '',    '',    '',    '',    '',    '',    ''],
    'UNK': ['N', 'CA', 'C', 'O', 'CB',   '',    '',    '',    '',    '',    '',    '',    '',    ''], # 给侧链原子

}

# {'resname':[['atomname',group,(coord)],
#             ['atomname',group,(coord)]] }
# group is the index of rigid group the atom belongs to, 8 (bb, psi, omega, phi, x1,x2,x3,x4)
# Notice that these positions are relative to the 'axis end atom'. e.g. for residue GLN,
# group 6 atom NE2 & OE1 are relative to atom CD (the origin). The X-axis is CD - CG, The
# Y-axis are placed so that OE1 - CD (the last entry in chi_angles_atoms)
# are in the X-Y plane and have a positive Y-axis.

# All the Oxygen atom are leave in group 3, the omega angle group, in case we change it latter

# The atom positions are relative to the axis-end-atom of the corresponding
# rotation axis. The x-axis is in direction of the rotation axis, and the y-axis
# is defined such that the dihedral-angle-definiting atom (the last entry in
# chi_angles_atoms above) is in the xy-plane (with a positive y-coordinate).
# format: [atomname, group_idx, rel_position]

residues_atom_position = {
    'ALA': [
        ['N', 0, (-0.525, 1.363, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, -0.000, -0.000)],
        ['CB', 0, (-0.529, -0.774, -1.205)],
        ['O', 3, (0.627, 1.062, 0.000)],
    ],
    'ARG': [
        ['N', 0, (-0.524, 1.362, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, -0.000, -0.000)],
        ['CB', 0, (-0.524, -0.778, -1.209)],
        ['O', 3, (0.626, 1.062, 0.000)],
        ['CG', 4, (0.616, 1.390, -0.000)],
        ['CD', 5, (0.564, 1.414, 0.000)],
        ['NE', 6, (0.539, 1.357, -0.000)],
        ['NH1', 7, (0.206, 2.301, 0.000)],
        ['NH2', 7, (2.078, 0.978, -0.000)],
        ['CZ', 7, (0.758, 1.093, -0.000)],
    ],
    'ASN': [
        ['N', 0, (-0.536, 1.357, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, -0.000, -0.000)],
        ['CB', 0, (-0.531, -0.787, -1.200)],
        ['O', 3, (0.625, 1.062, 0.000)],
        ['CG', 4, (0.584, 1.399, 0.000)],
        ['ND2', 5, (0.593, -1.188, 0.001)],
        ['OD1', 5, (0.633, 1.059, 0.000)],
    ],
    'ASP': [
        ['N', 0, (-0.525, 1.362, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.527, 0.000, -0.000)],
        ['CB', 0, (-0.526, -0.778, -1.208)],
        ['O', 3, (0.626, 1.062, -0.000)],
        ['CG', 4, (0.593, 1.398, -0.000)],
        ['OD1', 5, (0.610, 1.091, 0.000)],
        ['OD2', 5, (0.592, -1.101, -0.003)],
    ],
    'CYS': [
        ['N', 0, (-0.522, 1.362, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.524, 0.000, 0.000)],
        ['CB', 0, (-0.519, -0.773, -1.212)],
        ['O', 3, (0.625, 1.062, -0.000)],
        ['SG', 4, (0.728, 1.653, 0.000)],
    ],
    'GLN': [
        ['N', 0, (-0.526, 1.361, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, 0.000, 0.000)],
        ['CB', 0, (-0.525, -0.779, -1.207)],
        ['O', 3, (0.626, 1.062, -0.000)],
        ['CG', 4, (0.615, 1.393, 0.000)],
        ['CD', 5, (0.587, 1.399, -0.000)],
        ['NE2', 6, (0.593, -1.189, -0.001)],
        ['OE1', 6, (0.634, 1.060, 0.000)],
    ],
    'GLU': [
        ['N', 0, (-0.528, 1.361, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, -0.000, -0.000)],
        ['CB', 0, (-0.526, -0.781, -1.207)],
        ['O', 3, (0.626, 1.062, 0.000)],
        ['CG', 4, (0.615, 1.392, 0.000)],
        ['CD', 5, (0.600, 1.397, 0.000)],
        ['OE1', 6, (0.607, 1.095, -0.000)],
        ['OE2', 6, (0.589, -1.104, -0.001)],
    ],
    'GLY': [
        ['N', 0, (-0.572, 1.337, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.517, -0.000, -0.000)],
        ['O', 3, (0.626, 1.062, -0.000)],
    ],
    'HIS': [
        ['N', 0, (-0.527, 1.360, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, 0.000, 0.000)],
        ['CB', 0, (-0.525, -0.778, -1.208)],
        ['O', 3, (0.625, 1.063, 0.000)],
        ['CG', 4, (0.600, 1.370, -0.000)],
        ['CD2', 5, (0.889, -1.021, 0.003)],
        ['ND1', 5, (0.744, 1.160, -0.000)],
        ['CE1', 5, (2.030, 0.851, 0.002)],
        ['NE2', 5, (2.145, -0.466, 0.004)],
    ],
    'ILE': [
        ['N', 0, (-0.493, 1.373, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.527, -0.000, -0.000)],
        ['CB', 0, (-0.536, -0.793, -1.213)],
        ['O', 3, (0.627, 1.062, -0.000)],
        ['CG1', 4, (0.534, 1.437, -0.000)],
        ['CG2', 4, (0.540, -0.785, -1.199)],
        ['CD1', 5, (0.619, 1.391, 0.000)],
    ],
    'LEU': [
        ['N', 0, (-0.520, 1.363, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, -0.000, -0.000)],
        ['CB', 0, (-0.522, -0.773, -1.214)],
        ['O', 3, (0.625, 1.063, -0.000)],
        ['CG', 4, (0.678, 1.371, 0.000)],
        ['CD1', 5, (0.530, 1.430, -0.000)],
        ['CD2', 5, (0.535, -0.774, 1.200)],
    ],
    'LYS': [
        ['N', 0, (-0.526, 1.362, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, 0.000, 0.000)],
        ['CB', 0, (-0.524, -0.778, -1.208)],
        ['O', 3, (0.626, 1.062, -0.000)],
        ['CG', 4, (0.619, 1.390, 0.000)],
        ['CD', 5, (0.559, 1.417, 0.000)],
        ['CE', 6, (0.560, 1.416, 0.000)],
        ['NZ', 7, (0.554, 1.387, 0.000)],
    ],
    'MET': [
        ['N', 0, (-0.521, 1.364, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, 0.000, 0.000)],
        ['CB', 0, (-0.523, -0.776, -1.210)],
        ['O', 3, (0.625, 1.062, -0.000)],
        ['CG', 4, (0.613, 1.391, -0.000)],
        ['SD', 5, (0.703, 1.695, 0.000)],
        ['CE', 6, (0.320, 1.786, -0.000)],
    ],
    'PHE': [
        ['N', 0, (-0.518, 1.363, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.524, 0.000, -0.000)],
        ['CB', 0, (-0.525, -0.776, -1.212)],
        ['O', 3, (0.626, 1.062, -0.000)],
        ['CG', 4, (0.607, 1.377, 0.000)],
        ['CD1', 5, (0.709, 1.195, -0.000)],
        ['CD2', 5, (0.706, -1.196, 0.000)],
        ['CE1', 5, (2.102, 1.198, -0.000)],
        ['CE2', 5, (2.098, -1.201, -0.000)],
        ['CZ', 5, (2.794, -0.003, -0.001)],
    ],
    'PRO': [ # We set all the PRO atom into one rigid group which leave the chi angle constant
        ['N', 0, (-0.566, 1.351, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.527, -0.000, 0.000)],
        ['CB', 0, (-0.546, -0.611, -1.293)],
        ['O', 3, (0.621, 1.066, 0.000)],
        ['CG', 0, (0.382, 1.445, 0.0)], # from group 5 to group 0
        ['CD', 0, (0.427, 1.440, 0.0)], # from group 5 to group 0, using the correct angle.
        #['CD', 5, (0.477, 1.424, 0.0)],  # manually made angle 2 degrees larger
    ],
    'SER': [
        ['N', 0, (-0.529, 1.360, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, -0.000, -0.000)],
        ['CB', 0, (-0.518, -0.777, -1.211)],
        ['O', 3, (0.626, 1.062, -0.000)],
        ['OG', 4, (0.503, 1.325, 0.000)],
    ],
    'THR': [
        ['N', 0, (-0.517, 1.364, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, 0.000, -0.000)],
        ['CB', 0, (-0.516, -0.793, -1.215)],
        ['O', 3, (0.626, 1.062, 0.000)],
        ['CG2', 4, (0.550, -0.718, -1.228)],
        ['OG1', 4, (0.472, 1.353, 0.000)],
    ],
    'TRP': [
        ['N', 0, (-0.521, 1.363, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, -0.000, 0.000)],
        ['CB', 0, (-0.523, -0.776, -1.212)],
        ['O', 3, (0.627, 1.062, 0.000)],
        ['CG', 4, (0.609, 1.370, -0.000)],
        ['CD1', 5, (0.824, 1.091, 0.000)],
        ['CD2', 5, (0.854, -1.148, -0.005)],
        ['CE2', 5, (2.186, -0.678, -0.007)],
        ['CE3', 5, (0.622, -2.530, -0.007)],
        ['NE1', 5, (2.140, 0.690, -0.004)],
        ['CH2', 5, (3.028, -2.890, -0.013)],
        ['CZ2', 5, (3.283, -1.543, -0.011)],
        ['CZ3', 5, (1.715, -3.389, -0.011)],
    ],
    'TYR': [
        ['N', 0, (-0.522, 1.362, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.524, -0.000, -0.000)],
        ['CB', 0, (-0.522, -0.776, -1.213)],
        ['O', 3, (0.627, 1.062, -0.000)],
        ['CG', 4, (0.607, 1.382, -0.000)],
        ['CD1', 5, (0.716, 1.195, -0.000)],
        ['CD2', 5, (0.713, -1.194, -0.001)],
        ['CE1', 5, (2.107, 1.200, -0.002)],
        ['CE2', 5, (2.104, -1.201, -0.003)],
        ['OH', 5, (4.168, -0.002, -0.005)],
        ['CZ', 5, (2.791, -0.001, -0.003)],
    ],
    'VAL': [
        ['N', 0, (-0.494, 1.373, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.527, -0.000, -0.000)],
        ['CB', 0, (-0.533, -0.795, -1.213)],
        ['O', 3, (0.627, 1.062, -0.000)],
        ['CG1', 4, (0.540, 1.429, -0.000)],
        ['CG2', 4, (0.533, -0.776, 1.203)],
    ],
    'UNK': [ # Adding Atom for UNK residues, as UNK can be different res, here
             # I am naively using ALA's atom position. Notice that this can be wrong
             # when we Modify the Main chain position
             # 算主链的时候要注意这个，UNK 用不同的氨基酸原子位置可能导致其他res的位置不准确
        ['N', 0, (-0.525, 1.363, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, -0.000, -0.000)],
        ['CB', 0, (-0.529, -0.774, -1.205)],
        ['O', 3, (0.627, 1.062, 0.000)],
    ],
}

# Mask the useless chi angle for each residue. 1 means used, 0 means useless.
chi_angles_mask = [
    [0.0, 0.0, 0.0, 0.0],  # ALA
    [1.0, 1.0, 1.0, 1.0],  # ARG
    [1.0, 1.0, 0.0, 0.0],  # ASN
    [1.0, 1.0, 0.0, 0.0],  # ASP
    [1.0, 0.0, 0.0, 0.0],  # CYS
    [1.0, 1.0, 1.0, 0.0],  # GLN
    [1.0, 1.0, 1.0, 0.0],  # GLU
    [0.0, 0.0, 0.0, 0.0],  # GLY
    [1.0, 1.0, 0.0, 0.0],  # HIS
    [1.0, 1.0, 0.0, 0.0],  # ILE
    [1.0, 1.0, 0.0, 0.0],  # LEU
    [1.0, 1.0, 1.0, 1.0],  # LYS
    [1.0, 1.0, 1.0, 0.0],  # MET
    [1.0, 1.0, 0.0, 0.0],  # PHE
    [1.0, 1.0, 0.0, 0.0],  # PRO
    [1.0, 0.0, 0.0, 0.0],  # SER
    [1.0, 0.0, 0.0, 0.0],  # THR
    [1.0, 1.0, 0.0, 0.0],  # TRP
    [1.0, 1.0, 0.0, 0.0],  # TYR
    [1.0, 0.0, 0.0, 0.0],  # VAL
    [0.0, 0.0, 0.0, 0.0],  # UNK 暂且先不给UNK 安排任何chi angle
]
#
restype_frame_mask = [
    [1.0, 0.0, 0.0, 0.0, 0.0],  # ALA
    [1.0, 1.0, 1.0, 1.0, 1.0],  # ARG
    [1.0, 1.0, 1.0, 0.0, 0.0],  # ASN
    [1.0, 1.0, 1.0, 0.0, 0.0],  # ASP
    [1.0, 1.0, 0.0, 0.0, 0.0],  # CYS
    [1.0, 1.0, 1.0, 1.0, 0.0],  # GLN
    [1.0, 1.0, 1.0, 1.0, 0.0],  # GLU
    [1.0, 0.0, 0.0, 0.0, 0.0],  # GLY
    [1.0, 1.0, 1.0, 0.0, 0.0],  # HIS
    [1.0, 1.0, 1.0, 0.0, 0.0],  # ILE
    [1.0, 1.0, 1.0, 0.0, 0.0],  # LEU
    [1.0, 1.0, 1.0, 1.0, 1.0],  # LYS
    [1.0, 1.0, 1.0, 1.0, 0.0],  # MET
    [1.0, 1.0, 1.0, 0.0, 0.0],  # PHE
    [1.0, 0.0, 0.0, 0.0, 0.0],  # PRO # only PRO different from Alphafold chi_angles_mask
    [1.0, 1.0, 0.0, 0.0, 0.0],  # SER
    [1.0, 1.0, 0.0, 0.0, 0.0],  # THR
    [1.0, 1.0, 1.0, 0.0, 0.0],  # TRP
    [1.0, 1.0, 1.0, 0.0, 0.0],  # TYR
    [1.0, 1.0, 0.0, 0.0, 0.0],  # VAL
    [1.0, 0.0, 0.0, 0.0, 0.0],  # UNK 只有主链
]

# The following chi angles are pi periodic: they can be rotated by a multiple
# of pi without affecting the structure.
chi_pi_periodic = [
    [0.0, 0.0, 0.0, 0.0],  # ALA
    [0.0, 0.0, 0.0, 0.0],  # ARG
    [0.0, 0.0, 0.0, 0.0],  # ASN
    [0.0, 1.0, 0.0, 0.0],  # ASP
    [0.0, 0.0, 0.0, 0.0],  # CYS
    [0.0, 0.0, 0.0, 0.0],  # GLN
    [0.0, 0.0, 1.0, 0.0],  # GLU
    [0.0, 0.0, 0.0, 0.0],  # GLY
    [0.0, 0.0, 0.0, 0.0],  # HIS
    [0.0, 0.0, 0.0, 0.0],  # ILE
    [0.0, 0.0, 0.0, 0.0],  # LEU
    [0.0, 0.0, 0.0, 0.0],  # LYS
    [0.0, 0.0, 0.0, 0.0],  # MET
    [0.0, 1.0, 0.0, 0.0],  # PHE
    [0.0, 0.0, 0.0, 0.0],  # PRO
    [0.0, 0.0, 0.0, 0.0],  # SER
    [0.0, 0.0, 0.0, 0.0],  # THR
    [0.0, 0.0, 0.0, 0.0],  # TRP
    [0.0, 1.0, 0.0, 0.0],  # TYR
    [0.0, 0.0, 0.0, 0.0],  # VAL
    [0.0, 0.0, 0.0, 0.0],  # UNK
]


# Format: The list for each AA type contains chi1, chi2, chi3, chi4 in
# this order (or a relevant subset from chi1 onwards). ALA and GLY don't have
# chi angles so their chi angle lists are empty.
chi_angles_atoms = {
    'ALA': [],
    # Chi5 in arginine is always 0 +- 5 degrees, so ignore it.
    'ARG': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'NE'], ['CG', 'CD', 'NE', 'CZ']],
    'ASN': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']],
    'ASP': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']],
    'CYS': [['N', 'CA', 'CB', 'SG']],
    'GLN': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'OE1']],
    'GLU': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'OE1']],
    'GLY': [],
    'HIS': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'ND1']],
    'ILE': [['N', 'CA', 'CB', 'CG1'], ['CA', 'CB', 'CG1', 'CD1']],
    'LEU': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'LYS': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'],
            ['CB', 'CG', 'CD', 'CE'], ['CG', 'CD', 'CE', 'NZ']],
    'MET': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'SD'],
            ['CB', 'CG', 'SD', 'CE']],
    'PHE': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'PRO': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD']],
    'SER': [['N', 'CA', 'CB', 'OG']],
    'THR': [['N', 'CA', 'CB', 'OG1']],
    'TRP': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'TYR': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
    'VAL': [['N', 'CA', 'CB', 'CG1']],
    'UNK': [],
}

# This mapping is used when we need to store atom data in a format that requires
# fixed atom data size for every residue (e.g. a numpy array).
atom_types = [
    "N",
    "CA",
    "C",
    "CB",
    "O",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
]
atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}


# create an array with (restype, atomtype) --> rigid_group_idx
# and an array with (restype, atomtype, coord) for the atom positions
# and compute affine transformation matrices (4,4) from one rigid group to the
# previous group
restype_atom14_to_rigid_group = np.zeros([21, 14], dtype=int)
restype_atom14_mask = np.zeros([21, 14], dtype=np.float32)
restype_atom14_rigid_group_positions = np.zeros([21, 14, 3], dtype=np.float32)
restype_rigid_group_default_frame = np.zeros([21, 8, 4, 4], dtype=np.float32)
restype_atom37_mask = np.zeros([21, 37], dtype=np.float32)

def make_rigid_trans(ex, y_vec, t):
    """Create rigid rotation and translation matrix with the given axis and translation vec
    Return a  4x4 numpy array"""

    ex_norm = ex/ np.linalg.norm(ex)

    ey = y_vec - np.dot(y_vec, ex_norm) * ex_norm
    ey_norm = ey/ np.linalg.norm(ey)

    ez_norm = np.cross(ex_norm, ey_norm)

    m = np.stack([ex_norm, ey_norm, ez_norm, t]).transpose()

    m = np.concatenate([m, [[0.0, 0.0, 0.0, 1.0]]], axis=0)

    return m


def _make_rigid_group_constants():
    """Make rigid frames separated by torsion angles for each residue."""
    
    for residx, restype1 in enumerate(restypes):
        restype3 = restype_1to3[restype1]
        res_atom_position = {name: np.array(pos) for name, _, pos in residues_atom_position[restype3]}

        for atomname, group_idx, atom_position in residues_atom_position[restype3]:

            atomtype = atom_order[atomname]
            restype_atom37_mask[residx, atomtype] = 1

            atom14idx = restype_name_to_atom14_names[restype3].index(atomname)
            restype_atom14_to_rigid_group[residx, atom14idx] = group_idx
            restype_atom14_mask[residx, atom14idx] = 1
            restype_atom14_rigid_group_positions[residx, atom14idx, :] = atom_position


        # backbone to backbone is the identity transform
        restype_rigid_group_default_frame[residx, 0, :, :] = np.eye(4)


        # pre-omega-frame to backbone (currently dummy identity matrix)
        restype_rigid_group_default_frame[residx, 1, :, :] = np.eye(4)
        restype_rigid_group_default_frame[residx, 2, :, :] = np.eye(4)
        restype_rigid_group_default_frame[residx, 3, :, :] = np.eye(4)


        '''
        主链原子坐标暂时不由frame计算，因为这种计算方式忽略了肽键的存在，
        # phi-frame to backbone
        mat = make_rigid_trans(
            ex=res_atom_position["N"] - res_atom_position["CA"],
            ey=np.array([1.0, 0.0, 0.0]),
            translation=res_atom_position["N"],
        )
        restype_rigid_group_default_frame[residx, 2, :, :] = mat

        # psi-frame to backbone
        mat = make_rigid_trans(
            ex=res_atom_position["C"] - res_atom_position["CA"],
            ey=res_atom_position["CA"] - res_atom_position["N"],
            translation=res_atom_position["C"],
        )
        restype_rigid_group_default_frame[residx, 3, :, :] = mat
        '''

        # chi1-frame to backbone
        if chi_angles_mask[residx][0]:
            base_atom_names = chi_angles_atoms[restype3][0]
            base_atom_positions = [
                res_atom_position[name] for name in base_atom_names
            ]
            mat = make_rigid_trans(
                ex=base_atom_positions[2] - base_atom_positions[1],
                y_vec=base_atom_positions[0] - base_atom_positions[1],
                t=base_atom_positions[2],
            )
            restype_rigid_group_default_frame[residx, 4, :, :] = mat

        # chi2-frame to chi1-frame
        # chi3-frame to chi2-frame
        # chi4-frame to chi3-frame
        # luckily all rotation axes for the next frame start at (0,0,0) of the
        # previous frame
        for chi_idx in range(1, 4):
            if chi_angles_mask[residx][chi_idx]:
                axis_end_atom_name = chi_angles_atoms[restype3][chi_idx][2]
                axis_end_atom_position = res_atom_position[axis_end_atom_name]
                mat = make_rigid_trans(
                    ex=axis_end_atom_position,
                    y_vec=np.array([-1.0, 0.0, 0.0]),
                    t=axis_end_atom_position,
                )
                restype_rigid_group_default_frame[
                    residx, 4 + chi_idx, :, :
                ] = mat

def make_atom14_37_list():
    
    restype_atom14_to_atom37 = []
    restype_atom37_to_atom14 = []

    for rt in restypes:
        atom_names = restype_name_to_atom14_names[restype_1to3[rt]]

        restype_atom14_to_atom37.append([(atom_order[name] if name else 0)
                                         for name in atom_names])

        atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
        restype_atom37_to_atom14.append([
            (atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0)
            for name in atom_types
        ])
        
    restype_atom37_to_atom14 = np.array(restype_atom37_to_atom14)
        
    restype_atom37_to_atom14 = torch.tensor(restype_atom37_to_atom14)
    
    return restype_atom37_to_atom14

_make_rigid_group_constants()


