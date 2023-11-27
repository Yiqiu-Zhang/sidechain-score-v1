#!/bin/bash
#SBATCH --job-name=Sample
#SBATCH --output=output_Sample_graphIPA.log
#SBATCH --error=error_Sample_graphIPA.log
python sample.py -m result_graphIPA_m -o sample_graphIPA