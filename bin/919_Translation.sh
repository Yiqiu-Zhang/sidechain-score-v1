#!/bin/bash
#SBATCH --job-name=919_Translation
#SBATCH --output=output_919_Translation_new.log
#SBATCH --error=error_919_Translation_new.log
python train.py /mnt/petrelfs/zhangyiqiu/sidechain-score-v1/config_jsons/cath_full_angles_cosine-1.json --dryrun -o result_919_Translation_new