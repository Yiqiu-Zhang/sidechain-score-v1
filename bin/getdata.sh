#!/bin/bash
#SBATCH --job-name=getdata
#SBATCH --output=output_getdata.log
#SBATCH --error=error_getdata.log
python train.py /mnt/petrelfs/zhangyiqiu/sidechain-score-v1/config_jsons/getdata.json --dryrun -o getdata