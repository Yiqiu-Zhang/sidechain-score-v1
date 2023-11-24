#!/bin/bash
#SBATCH --job-name=924_Rigid_loc
#SBATCH --output=output_924_Rigid_loc.log
#SBATCH --error=error_924_Rigid_loc.log
python train.py /mnt/petrelfs/zhangyiqiu/sidechain-score-v1/config_jsons/cath_full_angles_cosine-1.json --dryrun -o result_924_Rigid_loc