#!/bin/bash
#SBATCH --job-name=test
#SBATCH -p bio_s1

### This needs to match Trainer(num_nodes=...)
#SBATCH --nodes=2

###单个节点使用的GPU个数，如果不需要GPU，则忽略该项
#SBATCH --gres=gpu:8

### This needs to match Trainer(devices=...)
#SBATCH --ntasks-per-node=8

#SBATCH --cpus-per-task=12
### request all the memory on a node
#SBATCH --mem=0

#SBATCH --output=out_test.log
#SBATCH --error=error_test.log

export NCCL_IB_DISABLE=1
export NCCL_IB_HCA=mlx5_0 
export NCCL_SOCKET_IFNAME=eth0
export CUDA_LAUNCH_BLOCKING=1

srun --kill-on-bad-exit=1 python3 train.py /mnt/petrelfs/zhangyiqiu/sidechain-score-v1/config_jsons/test.json --ndevice 8 --node 2 -o test