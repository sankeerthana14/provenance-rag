#!/bin/bash
#SBATCH --job-name=ablation_all
#SBATCH --partition=RTXA6Kq
#SBATCH --gres=gpu:1
#SBATCH --output=logs/ablation_all_%j.out
#SBATCH --error=logs/ablation_all_%j.err

export PYTHONUNBUFFERED=1

for feat in source_diversity text_resolution_rate avg_evidence_length min_evidence_length duplicate_rate document_overlap_rate entity_overlap evidence_count; do
    echo "========== STARTING ABLATION: ${feat} =========="
    python -u train_detector_v2.py --experiment roberta_feat \
        --train_path /export/home2/sati0004/TKDE/provenance-rag/provenance-rag/data/final_files/train.json \
        --val_path /export/home2/sati0004/TKDE/provenance-rag/provenance-rag/data/final_files/val.json \
        --test_path /export/home2/sati0004/TKDE/provenance-rag/provenance-rag/data/final_files/test.json \
        --model_dir /export/home2/sati0004/TKDE/provenance-rag/provenance-rag/models\
        --output_dir /export/home2/sati0004/TKDE/provenance-rag/provenance-rag/results/roberta_ablations/ablate_${feat} \
        --ablate_feature ${feat} \
        --batch_size 16 \
        --epochs 5
    echo "========== FINISHED ABLATION: ${feat} =========="
done