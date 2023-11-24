#!/bin/bash
#SBATCH --job-name=graphIPA
#SBATCH -p bio_s1
#SBATCH --gres=gpu:6
#SBATCH --ntasks-per-node=6
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --output=output_graphIPA.log
#SBATCH --error=error_graphIPA.log

### debugging flags (optional)
#export NCCL_DEBUG=INFO
#export PYTHONFAULTHANDLER=1

### on your cluster you might need these:
export NCCL_IB_HCA=mlx5_0
export NCCL_SOCKET_IFNAME=eth


srun --kill-on-bad-exit=1 python3 train.py /mnt/petrelfs/zhangyiqiu/sidechain-score-v1/config_jsons/graphIPA.json --ndevice 6 -o result_graphIPA