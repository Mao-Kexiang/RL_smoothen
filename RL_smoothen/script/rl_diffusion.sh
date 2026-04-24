#! /bin/bash
#SBATCH --partition=home    # 默认在home分区进行提交
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:H200:1
#SBATCH --time=1-00:00:00
#SBATCH --job-name=test
#SBATCH -o rl_diffusion.out

echo The current job ID is $SLURM_JOB_ID
echo List of CUDA devices: $CUDA_VISIBLE_DEVICES
echo

# 激活你的 conda 环境
source ~/.bashrc
conda activate RLtoy

# 进入你的项目目录
cd /home/user_milksang/private/homefile/RL_smoothen/toy_model

# 打印信息
echo "Job ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ==== Job started at `date` ====
echo ==== Current working directory is `pwd` ====
nvidia-smi

# 运行程序
  python -u exp_diffusion/run_diffusion.py \
    --n_steps 100 \
    --pretrain_method fm \
    --pretrain_epochs 2000 \
    --pretrain_batch 2048 \
    --dataset_size 100000 \
    --ppo_iters 300 \
    --ppo_batch 512
    
# 结束标记
echo "==== Job finished at $(date) ===="