#!/bin/bash
#SBATCH --job-name=927_DataNoise
#SBATCH --output=output_927_DataNoise.log
#SBATCH --error=error_927_DataNoise.log
python train.py /mnt/petrelfs/zhangyiqiu/sidechain-score-v1/config_jsons/cath_full_angles_cosine-1.json --dryrun -o result_927_DataNoise