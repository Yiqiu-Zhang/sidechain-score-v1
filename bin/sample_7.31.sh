#!/bin/bash
#SBATCH --job-name=sample_7.31
#SBATCH --output=output_sample_7.31.log
#SBATCH --error=error_sample_7.31.log
python sample.py -m resutl_IPA_Model_no_time_7_28  -o IPA_sample_no_time_7.31