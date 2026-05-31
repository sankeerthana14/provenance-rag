#!/bin/bash
#SBATCH --job-name=majority_expt
#SBATCH --gres=gpu:1
#SBATCH --partition=V100q
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module purge

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate prove-rag

python train_detector.py \
    --experiment majority \
    --train_path /export/home2/sati0004/TKDE/provenance-rag/provenance-rag/results/unified_train.json \
    --val_path /export/home2/sati0004/TKDE/provenance-rag/provenance-rag/results/unified_val.json \
    --output_dir /export/home2/sati0004/TKDE/provenance-rag/provenance-rag/results/majority_expt \
    --model_dir /export/home2/sati0004/TKDE/provenance-rag/provenance-rag/models 