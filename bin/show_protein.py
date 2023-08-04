import MDAnalysis as mda
from MDAnalysis.visualization import viewer

# 读入PDB文件
u = mda.Universe("protein.pdb")

# 显示分子3D结构
viewer.view(u)