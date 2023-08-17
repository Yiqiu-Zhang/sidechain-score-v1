#!/bin/bash
#SBATCH --job-name=sample_815
#SBATCH --output=output_sample_8_15.log
#SBATCH --error=error_sample_8_15.log
python sample.py -m resutl_periodic_neighbour_8_15 -o sample_periodic_neighbour_8_15