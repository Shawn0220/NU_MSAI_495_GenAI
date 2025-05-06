#!/bin/bash
#SBATCH --job-name=vae_hyperband
#SBATCH --output=logs/vae_output.log
#SBATCH --error=logs/vae_error.log
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=xiaoqin2026.1@u.northwestern.edu
#SBATCH --account=e32706
#SBATCH --hint=nomultithread
#SBATCH --chdir=/gpfs/home/qhm7800/MSAI337_NLP/vae_hyperband

# === 环境准备 ===
module purge
module load python-miniconda3/4.12.0
source /software/anaconda3/2018.12/etc/profile.d/conda.sh
conda activate /projects/e32706/qhm7800_envs/nlp_hw00  # 更换成你已有的或者新建的VAE环境

# === 检查 Python 环境 ===
which python
python --version
pip list | grep -E 'torch|ray|numpy'

# === 确保 logs 文件夹存在 ===
mkdir -p logs

# === 记录开始时间 ===
start_time=$(date +%s)

# === 运行 VAE 超参数搜索脚本 ===
python train_vae.py

# === 记录结束时间，计算耗时 ===
end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "Hyperparameter search completed in ${runtime} seconds."
