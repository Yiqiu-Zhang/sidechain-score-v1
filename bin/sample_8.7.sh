#!/bin/bash
#SBATCH --job-name=sample_8.7
#SBATCH --output=output_sample_8.7.log
#SBATCH --error=error_sample_8.7.log
python sample.py -m resutl_Score_8_3 -o IPA_sample_score_8.7