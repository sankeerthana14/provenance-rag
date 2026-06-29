"""
cross_dataset_eval.py
=====================
Cross-dataset generalization for PROVE-RAG.

Trains on 2 datasets, tests on the held-out 3rd.
Uses the best model (RoBERTa + Features, Approach B).

Three conditions:
  1. Train on FEVER+HotpotQA → Test on MuSiQue
  2. Train on FEVER+MuSiQue → Test on HotpotQA
  3. Train on HotpotQA+MuSiQue → Test on FEVER

Usage:
    python cross_dataset_eval.py \
        --data_dir /path/to/data/processed/ \
        --model_dir /path/to/models \
        --output_dir results/cross_dataset \
        --encoder_path models/roberta-base
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from itertools import combinations
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix
)

# Import from train_detector
sys.path.insert(0, os.path.dirname(__file__))
from train_detector import (
    DebertaWithFeatures, EvidenceDataset,
    LABEL2ID, ID2LABEL, NUM_LABELS,
    FEATURE_NAMES, NUM_FEATURES,
    load_data, evaluate_and_save,
    plot_confusion_matrix
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-dataset generalization"
    )
    parser.add_argument("--data_dir", type=str, required=True,
        help="Directory with train.json, val.json, test.json")
    parser.add_argument("--encoder_path", type=str, required=True,
        help="Path to RoBERTa-base model")
    parser.add_argument("--output_dir", type=str,
        default="results/cross_dataset")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_and_pool_all_data(data_dir):
    """Load train+val+test and pool all instances."""
    all_data = []
    for split in ["train.json", "val.json", "test.json"]:
        path = os.path.join(data_dir, split)
        if os.path.exists(path):
            print(f"  Loading {path}...")
            with open(path, "r") as f:
                data = json.load(f)
            all_data.extend(data)
            print(f"    {len(data)} instances")
    print(f"  Total pooled: {len(all_data)} instances")
    return all_data


def split_by_dataset(all_data):
    """Split pooled data by dataset field."""
    by_dataset = {}
    for inst in all_data:
        ds = inst.get("dataset", "unknown")
        if ds not in by_dataset:
            by_dataset[ds] = []
        by_dataset[ds].append(inst)

    for ds, instances in by_dataset.items():
        labels = Counter(
            inst["evidence_state_label"]
            for inst in instances
        )
        print(f"  {ds}: {len(instances)} instances, "
              f"labels={dict(labels)}")
    return by_dataset


def extract_from_instances(instances, use_structured=True):
    """Extract texts, labels, features, datasets."""
    texts, labels, features, datasets = [], [], [], []
    for inst in instances:
        # Text
        if use_structured and inst.get("structured_input"):
            text = inst["structured_input"]
        elif inst.get("plain_input"):
            text = inst["plain_input"]
        else:
            text = inst.get("input_text", "")
        texts.append(text)

        # Label
        labels.append(LABEL2ID[inst["evidence_state_label"]])

        # Features
        gf = inst.get("graph_features", {})
        feat = [float(gf.get(fn, 0.0))
                for fn in FEATURE_NAMES]
        features.append(feat)

        # Dataset
        datasets.append(inst.get("dataset", "unknown"))

    return texts, labels, features, datasets


def train_and_evaluate(
    train_texts, train_labels, train_features,
    test_texts, test_labels, test_features,
    test_datasets, encoder_path, output_dir, args,
    experiment_name
):
    """Train RoBERTa+Features and evaluate."""
    from transformers import AutoTokenizer

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Tokenize
    print(f"  Tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(
        encoder_path, use_fast=False
    )

    train_enc = tokenizer(
        [t[:5000] for t in train_texts],
        truncation=True, padding="max_length",
        max_length=args.max_length, return_tensors="pt"
    )
    test_enc = tokenizer(
        [t[:5000] for t in test_texts],
        truncation=True, padding="max_length",
        max_length=args.max_length, return_tensors="pt"
    )

    # Normalize features
    train_feat = np.array(train_features, dtype=np.float32)
    test_feat = np.array(test_features, dtype=np.float32)
    feat_mean = train_feat.mean(axis=0)
    feat_std = train_feat.std(axis=0) + 1e-8
    train_feat = (train_feat - feat_mean) / feat_std
    test_feat = (test_feat - feat_mean) / feat_std

    # Split train into train/val (90/10)
    n = len(train_labels)
    n_val = max(1, int(0.1 * n))
    indices = np.random.permutation(n)
    val_idx = indices[:n_val]
    tr_idx = indices[n_val:]

    # Create datasets
    train_dataset = EvidenceDataset(
        {k: v[tr_idx] for k, v in train_enc.items()},
        [train_labels[i] for i in tr_idx],
        train_feat[tr_idx].tolist()
    )
    val_dataset = EvidenceDataset(
        {k: v[val_idx] for k, v in train_enc.items()},
        [train_labels[i] for i in val_idx],
        train_feat[val_idx].tolist()
    )
    test_dataset = EvidenceDataset(
        test_enc, test_labels, test_feat.tolist()
    )

    # Model
    print(f"  Loading model...")
    model = DebertaWithFeatures(
        model_name=encoder_path,
        num_labels=NUM_LABELS,
        num_features=NUM_FEATURES,
        use_features=True,
    ).to(device).float()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size * 2,
        shuffle=False, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size * 2,
        shuffle=False, num_workers=0
    )

    # Scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, 1.0 - (step - warmup_steps)
                   / (total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda
    )

    # Train
    print(f"  Training for {args.epochs} epochs...")
    best_val_f1 = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            features = batch["features"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                features=features, labels=labels
            )
            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            n_batches += 1

        # Validate
        model.eval()
        val_preds, val_labels_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attn_mask = batch["attention_mask"].to(device)
                features = batch["features"].to(device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    features=features
                )
                preds = torch.argmax(
                    outputs["logits"], dim=-1
                )
                val_preds.extend(preds.cpu().tolist())
                val_labels_list.extend(
                    batch["labels"].tolist()
                )

        val_f1 = f1_score(
            val_labels_list, val_preds, average="macro"
        )
        print(f"    Epoch {epoch+1}: loss="
              f"{epoch_loss/n_batches:.4f}, "
              f"val_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, "best_model.pt")
            )

    # Load best and evaluate on test
    print(f"  Evaluating on held-out dataset...")
    model.load_state_dict(
        torch.load(os.path.join(output_dir, "best_model.pt"))
    )
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            features = batch["features"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                features=features
            )
            preds = torch.argmax(
                outputs["logits"], dim=-1
            )
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["labels"].tolist())

    evaluate_and_save(
        all_labels, all_preds, test_datasets,
        output_dir, experiment_name
    )

    # Clean up GPU
    del model
    torch.cuda.empty_cache()


def main():
    args = parse_args()

    print("=" * 60)
    print("PROVE-RAG Cross-Dataset Generalization")
    print("=" * 60)

    # Load and pool all data
    print("\n[1] Loading all data...")
    all_data = load_and_pool_all_data(args.data_dir)

    # Split by dataset
    print("\n[2] Splitting by dataset...")
    by_dataset = split_by_dataset(all_data)
    del all_data

    datasets = sorted(by_dataset.keys())
    print(f"  Datasets: {datasets}")

    # Run cross-dataset experiments
    print("\n[3] Running cross-dataset experiments...")

    results_summary = {}

    for held_out in datasets:
        train_datasets = [d for d in datasets
                         if d != held_out]
        exp_name = (f"Train({'+'.join(train_datasets)})"
                    f"_Test({held_out})")

        print(f"\n{'='*60}")
        print(f"  {exp_name}")
        print(f"{'='*60}")

        # Pool training data
        train_instances = []
        for d in train_datasets:
            train_instances.extend(by_dataset[d])
        test_instances = by_dataset[held_out]

        print(f"  Train: {len(train_instances)} "
              f"from {train_datasets}")
        print(f"  Test: {len(test_instances)} "
              f"from [{held_out}]")

        # Extract
        (train_texts, train_labels,
         train_features, train_ds) = \
            extract_from_instances(train_instances)
        (test_texts, test_labels,
         test_features, test_ds) = \
            extract_from_instances(test_instances)

        # Train and evaluate
        exp_output = os.path.join(
            args.output_dir, f"test_{held_out}"
        )
        train_and_evaluate(
            train_texts, train_labels, train_features,
            test_texts, test_labels, test_features,
            test_ds, args.encoder_path, exp_output,
            args, exp_name
        )

    print(f"\n{'='*60}")
    print("DONE! All cross-dataset experiments complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()