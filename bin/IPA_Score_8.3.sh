#!/bin/bash
#SBATCH --job-name=8_3_score_ipa
#SBATCH --output=output_score_ipa_8_3
#SBATCH --error=error_score_ipa_8_3
python train.py /mnt/petrelfs/lvying/code/sidechain_score/config_jsons/cath_full_angles_cosine-1.json --dryrun -o resutl_Score_8_3
