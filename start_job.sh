#!/bin/bash

#SBATCH --partition=gpuexpress
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-01:00:00
#SBATCH --job-name=conf_report
#SBATCH --output=slurm_logs/conf_report.dat
#SBATCH --error=slurm_logs/conf_report_err.dat 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=e_dehn01@uni-muenster.de

echo "###############################################################################"
echo "# Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"

module load palma/2024a foss/2024a Python/3.12.3 CUDA/12.6.0

pip install --user uv

export UV_LINK_MODE=copy

cd /scratch/tmp/e_dehn01/activation-patching

uv sync --frozen

uv run python src/report_confirmatory.py

echo "# Job finished at $(date)"