#!/bin/bash
#SBATCH --job-name=Angle_IPA_Model
#SBATCH --output=output_IPA_Model_6_15.log
#SBATCH --error=error_IPA_Model_6_15.log
python train.py /mnt/petrelfs/lvying/code/sidechain-rigid-attention/config_jsons/cath_full_angles_cosine.json --dryrun -o resutl_IPA_Model_6_15