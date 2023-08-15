#!/bin/bash
#SBATCH --job-name=8_3_score_ipa
#SBATCH --output=output_score_ipa_8_3.log
#SBATCH --error=error_score_ipa_8_3.log
python train.py /mnt/petrelfs/zhangyiqiu/sidechain-score-v1/config_jsons/cath_full_angles_cosine-1.json --dryrun -o resutl_score_up_neighbour_8_11