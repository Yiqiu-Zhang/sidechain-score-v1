#!/bin/bash
### sbatch demobatch.sh &
#SBATCH -J train
###指定任务队列，默认使用normal
#SBATCH -p bio_s1
###申请2个节点，单节点任务可以省略不写
#SBATCH -N 2
###每个节点分配16个MPI进程，即cpu核心数。
#SBATCH --ntasks-per-node=16
###一共跑16个进程。也可以省略“-N”和” --ntasks-per-node”参数，这样任务会优先填满一个节点，不会平均分散在几个节点上。
#SBATCH -n 32
#SBATCH --mem=512G
#SBATCH -x SH-IDC1-10-140-1-31
#SBATCH --cpus-per-task=4
###每个节点申请一个gpu，cpu任务请注释掉
#SBATCH --gres=gpu:8
#SBATCH --get-user-env
###脚本错误输出
#SBATCH -e job-%j.err
###脚本输出
#SBATCH -o job-%j.out
###进入提交任务时所在目录

###下面是要执行的程序
echo "The nodes allocated to the job is $SLURM_JOB_NODELIST ."
echo "The number of processors on this node allocated to the job is $SLURM_JOB_CPUS_PER_NODE ."
echo "Number of tasks to be initiated on each node is $SLURM_TASKS_PER_NODE ."

date -R

source ~/.bashrc
conda activate tf2.6
cd $SLURM_SUBMIT_DIR
ulimit -s unlimited

export NODELIST=nodelist.$
srun -l bash -c 'hostname' |  sort -k 2 -u | awk -vORS=, '{print $2":4"}' | sed 's/,$//' > $NODELIST

## Number of total processes
echo " "
echo " Nodelist:= " $SLURM_JOB_NODELIST
echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
echo " GPUs per node:= " $SLURM_JOB_GPUS
echo " Ntasks per node:= "  $SLURM_NTASKS_PER_NODE

#mpirun不加参数会默认以80个进程运行
mpirun --host `cat $NODELIST` -np 32 -bind-to none -map-by slot python -u train_horovod.py

echo Completed!

