#! /bin/bash
#SBATCH --partition=home    # 默认在home分区进行提交
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:A100_40G:1
#SBATCH --time=1-00:00:00
#SBATCH --job-name=test
#SBATCH -o rl_flow.out

echo The current job ID is $SLURM_JOB_ID
echo List of CUDA devices: $CUDA_VISIBLE_DEVICES
echo

# 激活你的 conda 环境
source ~/.bashrc
conda activate RLtoy

# 进入你的项目目录
cd /home/user_milksang/private/homefile/RL_smoothen/toy_model
mkdir -p output/flow

# 打印信息
echo "Job ID: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ==== Job started at `date` ====
echo ==== Current working directory is `pwd` ====
nvidia-smi

# 运行程序
  # python -u exp_2d/model.py --pretrain gaussian --gaussian_std 0.5 --ppo_kl 0.0 --dataset_size 100000 --scan
  python -u exp_2d/model.py --pretrain boltzmann --beta 1.0 --hidden_dim 64 --ppo_kl 0.0 --ppo_iters 200 --ppo_batch 1024 --dataset_size 100000 --init kaiming --n_layers 8 --pretrain_epochs 100000 --pretrain_lr 1e-5  --pretrain_batch 1024
  # 使用 float64 精度：加 --dtype float64
    
# 结束标记
echo "==== Job finished at $(date) ===="