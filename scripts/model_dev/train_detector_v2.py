"""
train_detector.py
==================
Train and evaluate evidence-state detectors for PROVE-RAG.

Supports all Tier 1 experiments:
    --experiment majority        : Majority class baseline
    --experiment tfidf           : TF-IDF + Logistic Regression
    --experiment deberta_text    : DeBERTa-v3-base text-only (Approach A)
    --experiment deberta_feat    : DeBERTa-v3-base + provenance features (Approach B)

Also supports Tier 2:
    --experiment bert_text       : BERT-base text-only
    --experiment roberta_text    : RoBERTa-base text-only
    --experiment features_only   : Logistic Regression on 8 features only

Usage:
    # Run all Tier 1 experiments
    python train_detector.py --experiment majority --train_path data/unified_train.json --val_path data/unified_val.json --output_dir results/majority
    python train_detector.py --experiment tfidf --train_path data/unified_train.json --val_path data/unified_val.json --output_dir results/tfidf
    python train_detector.py --experiment deberta_text --train_path data/unified_train.json --val_path data/unified_val.json --output_dir results/deberta_text
    python train_detector.py --experiment deberta_feat --train_path data/unified_train.json --val_path data/unified_val.json --output_dir results/deberta_feat

Requirements:
    pip install transformers datasets torch scikit-learn matplotlib seaborn accelerate
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from typing import Dict, List, Optional
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# CONFIGURATION
# ============================================================

LABEL2ID = {
    "sufficient": 0,
    "insufficient": 1,
    "contradicted": 2,
    "superseded": 3,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = 4

FEATURE_NAMES = [
    "source_diversity",
    "text_resolution_rate",
    "avg_evidence_length",
    "min_evidence_length",
    "duplicate_rate",
    "document_overlap_rate",
    "entity_overlap",
    "evidence_count",
]
NUM_FEATURES = len(FEATURE_NAMES)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train evidence-state detector for PROVE-RAG"
    )
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["majority", "tfidf", "features_only",
                                 "llm_zeroshot",
                                 "deberta_text", "deberta_feat",
                                 "bert_text", "bert_feat",
                                 "roberta_text", "roberta_feat",
                                 "deberta_large_text", "deberta_large_feat"],
                        help="Which experiment to run")
    parser.add_argument("--train_path", type=str, required=True,
                        help="Path to unified_train.json")
    parser.add_argument("--val_path", type=str, required=True,
                        help="Path to unified_val.json")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for model + results")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Max token length for transformer inputs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Training batch size")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Limit training samples (for debugging)")
    parser.add_argument("--max_val_samples", type=int, default=None,
                        help="Limit validation samples (for debugging)")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Local directory containing downloaded models (e.g. ./models)")
    parser.add_argument("--test_path", type=str, default=None,
                        help="Path to test.json (optional, for final evaluation)")
    parser.add_argument("--ablate_feature", type=str, default=None,
                        choices=FEATURE_NAMES,
                        help="Remove this feature for ablation study (use with deberta_feat)")
    parser.add_argument("--use_structured", action="store_true", default=True,
                        help="Use structured [SEP] input format (default: True)")
    parser.add_argument("--no_structured", action="store_true",
                        help="Disable structured input, use plain concatenation")
    parser.add_argument("--llm_model", type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct",
                        help="LLM model for zero-shot experiment")
    parser.add_argument("--max_llm_samples", type=int, default=2000,
                        help="Max samples for LLM zero-shot (LLM inference is slow)")
    return parser.parse_args()


# ============================================================
# DATA LOADING
# ============================================================

def load_data(path, max_samples=None, use_structured=True):
    """
    Load JSON and extract texts, labels, and features.
    
    Supports three formats:
    - v2 format: structured_input with [SEP] markers (from preprocess_v2.py)
    - Light format: pre-extracted evidence_text (from preprocess_light.py)
    - Full format: evidence_units list (from unified_train.json)
    """
    print(f"  Loading {path}...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
    
    print(f"  Loaded {len(data)} instances")
    
    texts = []
    labels = []
    features = []
    datasets_col = []
    
    for inst in data:
        # Priority: structured_input > evidence_text > evidence_units
        if use_structured and inst.get("structured_input"):
            combined = inst["structured_input"]
        elif inst.get("plain_input"):
            combined = inst["plain_input"]
        elif inst.get("evidence_text"):
            combined = inst["input_text"] + " " + inst["evidence_text"]
        else:
            input_text = inst["input_text"]
            evidence_texts = []
            for eu in inst.get("evidence_units", []):
                if eu.get("text") and eu["text"].strip():
                    evidence_texts.append(eu["text"].strip())
            combined = input_text + " " + " ".join(evidence_texts)
        
        texts.append(combined)
        
        # Label
        label_str = inst["evidence_state_label"]
        labels.append(LABEL2ID[label_str])
        
        # Graph features
        gf = inst.get("graph_features", {})
        feat_vec = [float(gf.get(fn, 0.0)) for fn in FEATURE_NAMES]
        features.append(feat_vec)
        
        # Dataset source
        datasets_col.append(inst.get("dataset", "unknown"))
    
    # Free memory
    del data
    
    print(f"  Class distribution: {Counter(labels)}")
    
    return texts, labels, features, datasets_col


# ============================================================
# EXPERIMENT 1: MAJORITY BASELINE
# ============================================================

def run_majority(train_labels, val_labels, val_datasets, output_dir):
    """Predict the most common training class for all val instances."""
    print("\n" + "=" * 60)
    print("EXPERIMENT: Majority Class Baseline")
    print("=" * 60)
    
    most_common = Counter(train_labels).most_common(1)[0][0]
    print(f"  Most common class: {ID2LABEL[most_common]}")
    
    preds = [most_common] * len(val_labels)
    
    evaluate_and_save(val_labels, preds, val_datasets, output_dir, "majority")


# ============================================================
# EXPERIMENT 2: TF-IDF + LOGISTIC REGRESSION
# ============================================================

def run_tfidf(train_texts, train_labels, val_texts, val_labels,
              val_datasets, output_dir):
    """TF-IDF features + Logistic Regression classifier."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    
    print("\n" + "=" * 60)
    print("EXPERIMENT: TF-IDF + Logistic Regression")
    print("=" * 60)
    
    # Truncate texts for TF-IDF (no need for full evidence)
    train_texts_trunc = [t[:2000] for t in train_texts]
    val_texts_trunc = [t[:2000] for t in val_texts]
    
    print("  Fitting TF-IDF + LR pipeline...")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2),
                                   sublinear_tf=True)),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                                    solver="lbfgs", multi_class="multinomial",
                                    random_state=42)),
    ])
    
    pipeline.fit(train_texts_trunc, train_labels)
    preds = pipeline.predict(val_texts_trunc).tolist()
    
    evaluate_and_save(val_labels, preds, val_datasets, output_dir, "tfidf")


# ============================================================
# EXPERIMENT 3: FEATURES-ONLY BASELINE
# ============================================================

def run_features_only(train_features, train_labels, val_features, val_labels,
                      val_datasets, output_dir):
    """Logistic Regression on the 8 provenance features only (no text)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    
    print("\n" + "=" * 60)
    print("EXPERIMENT: Features Only (Logistic Regression)")
    print("=" * 60)
    
    train_X = np.array(train_features)
    val_X = np.array(val_features)
    
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                                    solver="lbfgs", multi_class="multinomial",
                                    random_state=42)),
    ])
    
    pipeline.fit(train_X, train_labels)
    preds = pipeline.predict(val_X).tolist()
    
    evaluate_and_save(val_labels, preds, val_datasets, output_dir, "features_only")


# ============================================================
# EXPERIMENT: LLM ZERO-SHOT BASELINE
# ============================================================

def run_llm_zeroshot(val_texts, val_labels, val_datasets, output_dir, args):
    """
    Run an LLM in zero-shot mode to classify evidence states.
    Shows whether fine-tuning is needed or if an LLM can do this out-of-box.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer as CausalTokenizer
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: LLM Zero-Shot Classification")
    print(f"Model: {args.llm_model}")
    print(f"Max samples: {args.max_llm_samples}")
    print(f"{'='*60}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Subsample for speed (LLM inference is slow)
    max_samples = args.max_llm_samples or 2000
    if len(val_texts) > max_samples:
        import random
        rng = random.Random(args.seed)
        indices = rng.sample(range(len(val_texts)), max_samples)
        eval_texts = [val_texts[i] for i in indices]
        eval_labels = [val_labels[i] for i in indices]
        eval_datasets = [val_datasets[i] for i in indices]
        print(f"  Subsampled to {max_samples} examples")
    else:
        eval_texts = val_texts
        eval_labels = val_labels
        eval_datasets = val_datasets
    
    # Load model
    print(f"  Loading model {args.llm_model}...")
    
    # Resolve local path if model_dir is provided
    llm_path = args.llm_model
    if args.model_dir:
        local_path = os.path.join(args.model_dir, args.llm_model.split("/")[-1])
        if os.path.exists(local_path):
            llm_path = local_path
            print(f"  Using local model: {llm_path}")
    
    tokenizer = CausalTokenizer.from_pretrained(llm_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        llm_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Classification prompt template
    prompt_template = """You are an evidence quality assessor. Given a question and its supporting evidence, classify the evidence state into exactly one of these four categories:

- sufficient: The evidence fully supports answering the question
- insufficient: Critical evidence is missing to answer the question properly  
- contradicted: The evidence contains conflicting information
- superseded: The evidence contains outdated information that has been updated

Question and Evidence:
{text}

Respond with ONLY one word: sufficient, insufficient, contradicted, or superseded.

Classification:"""

    # Run inference
    print(f"  Running inference on {len(eval_texts)} examples...")
    preds = []
    
    label_map = {
        "sufficient": 0, "insufficient": 1,
        "contradicted": 2, "superseded": 3,
    }
    
    for i, text in enumerate(eval_texts):
        # Truncate text to fit in context window
        truncated_text = text[:2000]
        prompt = prompt_template.format(text=truncated_text)
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                          max_length=2048).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Decode only the generated tokens
        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip().lower()
        
        # Parse the response
        pred_label = parse_llm_response(generated)
        preds.append(pred_label)
        
        if (i + 1) % 100 == 0:
            # Calculate running accuracy
            correct = sum(1 for p, l in zip(preds, eval_labels[:len(preds)]) if p == l)
            running_acc = correct / len(preds)
            print(f"    Processed {i+1}/{len(eval_texts)} "
                  f"(running acc: {running_acc:.4f})")
    
    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()
    
    evaluate_and_save(eval_labels, preds, eval_datasets, output_dir,
                      f"LLM Zero-Shot ({args.llm_model.split('/')[-1]})")


def parse_llm_response(response):
    """Parse LLM output to extract predicted evidence state."""
    response = response.lower().strip().split("\n")[0].strip()
    
    # Direct match
    if "insufficient" in response:
        return 1
    elif "superseded" in response:
        return 3
    elif "contradicted" in response or "contradict" in response:
        return 2
    elif "sufficient" in response:
        return 0
    
    # If no match, default to most common confusion
    return 0  # Default to sufficient


# ============================================================
# EXPERIMENT 4 & 5: TRANSFORMER-BASED DETECTORS
# ============================================================

class DebertaWithFeatures(nn.Module):
    """
    DeBERTa (or any transformer) with provenance features
    concatenated to the CLS embedding before classification.
    
    Approach A (text-only): use_features=False
        CLS embedding (768) → classifier → 4 classes
    
    Approach B (text + features): use_features=True
        CLS embedding (768) + features (8) → classifier → 4 classes
    """
    
    def __init__(self, model_name, num_labels, num_features, use_features=False,
                 dropout=0.1):
        super().__init__()
        from transformers import AutoModel
        
        self.encoder = AutoModel.from_pretrained(model_name)
        self.use_features = use_features
        
        hidden_size = self.encoder.config.hidden_size
        classifier_input = hidden_size + (num_features if use_features else 0)
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels),
        )
        
        self.num_labels = num_labels
    
    def forward(self, input_ids, attention_mask, features=None, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # CLS token embedding
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch, hidden]
        cls_output = cls_output.float()  # Ensure float32 for classifier
        cls_output = self.dropout(cls_output)
        
        # Concatenate features if using Approach B
        if self.use_features and features is not None:
            features = features.float()  # Ensure float32
            cls_output = torch.cat([cls_output, features], dim=-1)
        
        logits = self.classifier(cls_output)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return {"loss": loss, "logits": logits}


class EvidenceDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for evidence-state detection."""
    
    def __init__(self, encodings, labels, features):
        self.encodings = encodings
        self.labels = labels
        self.features = features
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
        }
        return item


def run_transformer(train_texts, train_labels, train_features,
                    val_texts, val_labels, val_features,
                    val_datasets, output_dir, args,
                    model_name, use_features, experiment_name):
    """Train and evaluate a transformer-based detector."""
    from transformers import AutoTokenizer
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"Model: {model_name}")
    print(f"Features: {'Yes (Approach B)' if use_features else 'No (Approach A)'}")
    print(f"{'='*60}")
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    
    # Tokenize
    print("  Tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    
    # Truncate texts before tokenizing to save memory
    train_texts_trunc = [t[:5000] for t in train_texts]  # Pre-truncate long texts
    val_texts_trunc = [t[:5000] for t in val_texts]
    
    train_encodings = tokenizer(
        train_texts_trunc, truncation=True, padding="max_length",
        max_length=args.max_length, return_tensors="pt"
    )
    val_encodings = tokenizer(
        val_texts_trunc, truncation=True, padding="max_length",
        max_length=args.max_length, return_tensors="pt"
    )
    
    print(f"  Train: {len(train_labels)} samples")
    print(f"  Val: {len(val_labels)} samples")
    
    # Normalize features
    train_feat_np = np.array(train_features, dtype=np.float32)
    val_feat_np = np.array(val_features, dtype=np.float32)
    
    # Standardize using train statistics
    feat_mean = train_feat_np.mean(axis=0)
    feat_std = train_feat_np.std(axis=0) + 1e-8
    train_feat_np = (train_feat_np - feat_mean) / feat_std
    val_feat_np = (val_feat_np - feat_mean) / feat_std
    
    # Save normalization params for inference
    os.makedirs(output_dir, exist_ok=True)
    np.savez(os.path.join(output_dir, "feature_norm.npz"),
             mean=feat_mean, std=feat_std)
    
    # Create datasets
    train_dataset = EvidenceDataset(train_encodings, train_labels,
                                     train_feat_np.tolist())
    val_dataset = EvidenceDataset(val_encodings, val_labels,
                                   val_feat_np.tolist())
    
    # Create model
    print("  Loading model...")
    model = DebertaWithFeatures(
        model_name=model_name,
        num_labels=NUM_LABELS,
        num_features=NUM_FEATURES,
        use_features=use_features,
    ).to(device).float()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   weight_decay=0.01)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=0, pin_memory=True
    )
    
    # Learning rate scheduler with warmup
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, 1.0 - (step - warmup_steps) / (total_steps - warmup_steps))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    print(f"\n  Training for {args.epochs} epochs...")
    best_val_f1 = 0
    train_losses = []
    val_f1s = []
    
    for epoch in range(args.epochs):
        # Train
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            features = batch["features"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                features=features if use_features else None,
                labels=labels,
            )
            
            loss = outputs["loss"]
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % 500 == 0:
                avg_loss = epoch_loss / num_batches
                print(f"    Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Loss: {avg_loss:.4f}")
        
        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validate
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                features = batch["features"].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    features=features if use_features else None,
                )
                
                preds = torch.argmax(outputs["logits"], dim=-1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(batch["labels"].numpy().tolist())
        
        # Compute F1
        from sklearn.metrics import f1_score
        val_f1 = f1_score(all_labels, all_preds, average="macro")
        val_f1s.append(val_f1)
        
        print(f"  Epoch {epoch+1}/{args.epochs}: "
              f"Train Loss={avg_train_loss:.4f}, Val Macro-F1={val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            print(f"    ✓ New best model saved (F1={val_f1:.4f})")
    
    # Load best model and do final evaluation
    print(f"\n  Loading best model (F1={best_val_f1:.4f})...")
    model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pt")))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            features = batch["features"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                features=features if use_features else None,
            )
            
            preds = torch.argmax(outputs["logits"], dim=-1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(batch["labels"].numpy().tolist())
    
    # Plot training curves
    plot_training_curves(train_losses, val_f1s, output_dir, experiment_name)
    
    # Full evaluation
    evaluate_and_save(all_labels, all_preds, val_datasets, output_dir,
                      experiment_name)


# ============================================================
# EVALUATION
# ============================================================

def evaluate_and_save(true_labels, pred_labels, dataset_labels,
                      output_dir, experiment_name):
    """Compute all metrics and save results."""
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        classification_report, confusion_matrix
    )
    
    os.makedirs(output_dir, exist_ok=True)
    
    true_labels = list(true_labels)
    pred_labels = list(pred_labels)
    
    # Overall metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average="macro")
    macro_precision = precision_score(true_labels, pred_labels, average="macro")
    macro_recall = recall_score(true_labels, pred_labels, average="macro")
    
    print(f"\n  {'='*50}")
    print(f"  RESULTS: {experiment_name}")
    print(f"  {'='*50}")
    print(f"  Accuracy:        {accuracy:.4f}")
    print(f"  Macro-F1:        {macro_f1:.4f}")
    print(f"  Macro-Precision: {macro_precision:.4f}")
    print(f"  Macro-Recall:    {macro_recall:.4f}")
    
    # Per-class report
    label_names = [ID2LABEL[i] for i in range(NUM_LABELS)]
    report = classification_report(true_labels, pred_labels,
                                    target_names=label_names, digits=4)
    print(f"\n{report}")
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    plot_confusion_matrix(cm, label_names, output_dir, experiment_name)
    
    # Per-dataset breakdown
    if dataset_labels:
        print(f"\n  Per-dataset breakdown:")
        unique_datasets = sorted(set(dataset_labels))
        per_dataset_results = {}
        
        for ds in unique_datasets:
            ds_mask = [i for i, d in enumerate(dataset_labels) if d == ds]
            ds_true = [true_labels[i] for i in ds_mask]
            ds_pred = [pred_labels[i] for i in ds_mask]
            
            ds_acc = accuracy_score(ds_true, ds_pred)
            ds_f1 = f1_score(ds_true, ds_pred, average="macro")
            
            per_dataset_results[ds] = {"accuracy": ds_acc, "macro_f1": ds_f1,
                                        "n": len(ds_mask)}
            print(f"    {ds}: Acc={ds_acc:.4f}, F1={ds_f1:.4f} (n={len(ds_mask)})")
    
    # Action accuracy (evidence state → agentic action mapping)
    action_map = {0: "answer", 1: "retrieve_more", 2: "flag_conflict", 3: "verify"}
    true_actions = [action_map[l] for l in true_labels]
    pred_actions = [action_map[l] for l in pred_labels]
    action_acc = accuracy_score(true_actions, pred_actions)
    print(f"\n  Action Accuracy: {action_acc:.4f}")
    
    # Save all results
    results = {
        "experiment": experiment_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "action_accuracy": action_acc,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }
    
    if dataset_labels:
        results["per_dataset"] = per_dataset_results
    
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n  Results saved to {output_dir}/results.json")


def plot_confusion_matrix(cm, label_names, output_dir, experiment_name):
    """Plot and save confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"Confusion Matrix: {experiment_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.close()
    print(f"  Confusion matrix saved to {output_dir}/confusion_matrix.png")


def plot_training_curves(train_losses, val_f1s, output_dir, experiment_name):
    """Plot training loss and validation F1 curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(train_losses) + 1)
    
    ax1.plot(epochs, train_losses, "b-o", markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, val_f1s, "r-o", markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Macro-F1")
    ax2.set_title("Validation F1")
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(experiment_name, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=300)
    plt.close()


# ============================================================
# MODEL NAME MAPPING
# ============================================================

EXPERIMENT_CONFIG = {
    "deberta_text": {
        "model_name": "microsoft/deberta-v3-base",
        "local_folder": "deberta-v3-base",
        "use_features": False,
        "display_name": "DeBERTa-v3-base (text-only, Approach A)",
    },
    "deberta_feat": {
        "model_name": "microsoft/deberta-v3-base",
        "local_folder": "deberta-v3-base",
        "use_features": True,
        "display_name": "DeBERTa-v3-base + Features (Approach B)",
    },
    "bert_text": {
        "model_name": "bert-base-uncased",
        "local_folder": "bert-base-uncased",
        "use_features": False,
        "display_name": "BERT-base (text-only, Approach A)",
    },
    "bert_feat": {
        "model_name": "bert-base-uncased",
        "local_folder": "bert-base-uncased",
        "use_features": True,
        "display_name": "BERT-base + Features (Approach B)",
    },
    "roberta_text": {
        "model_name": "roberta-base",
        "local_folder": "roberta-base",
        "use_features": False,
        "display_name": "RoBERTa-base (text-only, Approach A)",
    },
    "roberta_feat": {
        "model_name": "roberta-base",
        "local_folder": "roberta-base",
        "use_features": True,
        "display_name": "RoBERTa-base + Features (Approach B)",
    },
    "deberta_large_text": {
        "model_name": "microsoft/deberta-v3-large",
        "local_folder": "deberta-v3-large",
        "use_features": False,
        "display_name": "DeBERTa-v3-large (text-only)",
    },
    "deberta_large_feat": {
        "model_name": "microsoft/deberta-v3-large",
        "local_folder": "deberta-v3-large",
        "use_features": True,
        "display_name": "DeBERTa-v3-large + Features",
    },
}


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()
    
    # Handle structured input flag
    use_structured = args.use_structured and not args.no_structured
    
    print("=" * 60)
    print("PROVE-RAG Evidence-State Detector Training")
    print(f"Experiment: {args.experiment}")
    print(f"Seed: {args.seed}")
    print(f"Structured input: {use_structured}")
    if args.ablate_feature:
        print(f"Ablating feature: {args.ablate_feature}")
    print("=" * 60)
    
    # Load data
    print("\n[1] Loading data...")
    train_texts, train_labels, train_features, train_datasets = load_data(
        args.train_path, args.max_train_samples, use_structured=use_structured
    )
    val_texts, val_labels, val_features, val_datasets = load_data(
        args.val_path, args.max_val_samples, use_structured=use_structured
    )
    
    # Apply feature ablation if requested
    if args.ablate_feature:
        feat_idx = FEATURE_NAMES.index(args.ablate_feature)
        print(f"\n  Ablating feature '{args.ablate_feature}' (index {feat_idx}) — setting to 0.0")
        for feat_vec in train_features:
            feat_vec[feat_idx] = 0.0
        for feat_vec in val_features:
            feat_vec[feat_idx] = 0.0
    
    # Run experiment
    print(f"\n[2] Running experiment: {args.experiment}")
    
    if args.experiment == "majority":
        run_majority(train_labels, val_labels, val_datasets, args.output_dir)
    
    elif args.experiment == "tfidf":
        run_tfidf(train_texts, train_labels, val_texts, val_labels,
                  val_datasets, args.output_dir)
    
    elif args.experiment == "features_only":
        run_features_only(train_features, train_labels, val_features, val_labels,
                          val_datasets, args.output_dir)
    
    elif args.experiment == "llm_zeroshot":
        run_llm_zeroshot(val_texts, val_labels, val_datasets, args.output_dir, args)
    
    elif args.experiment in EXPERIMENT_CONFIG:
        config = EXPERIMENT_CONFIG[args.experiment]
        
        # Resolve model path: use local directory if provided, else HuggingFace
        if args.model_dir:
            model_path = os.path.join(args.model_dir, config["local_folder"])
            if not os.path.exists(model_path):
                print(f"  WARNING: Local model not found at {model_path}")
                print(f"  Falling back to HuggingFace: {config['model_name']}")
                model_path = config["model_name"]
            else:
                print(f"  Using local model: {model_path}")
        else:
            model_path = config["model_name"]
        
        run_transformer(
            train_texts, train_labels, train_features,
            val_texts, val_labels, val_features,
            val_datasets, args.output_dir, args,
            model_name=model_path,
            use_features=config["use_features"],
            experiment_name=config["display_name"],
        )
    
    # Evaluate on test set if provided
    if args.test_path:
        print(f"\n{'='*60}")
        print("EVALUATING ON HELD-OUT TEST SET")
        print(f"{'='*60}")
        
        test_texts, test_labels, test_features, test_datasets = load_data(
            args.test_path, use_structured=use_structured
        )
        
        if args.ablate_feature:
            feat_idx = FEATURE_NAMES.index(args.ablate_feature)
            for feat_vec in test_features:
                feat_vec[feat_idx] = 0.0
        
        if args.experiment in EXPERIMENT_CONFIG:
            config = EXPERIMENT_CONFIG[args.experiment]
            # Load best model and evaluate on test
            import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            model_path_resolved = model_path if 'model_path' in dir() else config["model_name"]
            
            model = DebertaWithFeatures(
                model_name=model_path_resolved,
                num_labels=NUM_LABELS,
                num_features=NUM_FEATURES,
                use_features=config["use_features"],
            ).to(device).float()
            
            best_model_path = os.path.join(args.output_dir, "best_model.pt")
            if os.path.exists(best_model_path):
                model.load_state_dict(torch.load(best_model_path, map_location=device))
                model.eval()
                
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_path_resolved, use_fast=False)
                
                test_texts_trunc = [t[:5000] for t in test_texts]
                test_encodings = tokenizer(
                    test_texts_trunc, truncation=True, padding="max_length",
                    max_length=args.max_length, return_tensors="pt"
                )
                
                # Normalize features using saved params
                norm_path = os.path.join(args.output_dir, "feature_norm.npz")
                if os.path.exists(norm_path):
                    norm = np.load(norm_path)
                    test_feat_np = np.array(test_features, dtype=np.float32)
                    test_feat_np = (test_feat_np - norm["mean"]) / norm["std"]
                else:
                    test_feat_np = np.array(test_features, dtype=np.float32)
                
                test_dataset = EvidenceDataset(test_encodings, test_labels,
                                                test_feat_np.tolist())
                test_loader = torch.utils.data.DataLoader(
                    test_dataset, batch_size=args.batch_size * 2, shuffle=False
                )
                
                all_preds = []
                all_labels_test = []
                with torch.no_grad():
                    for batch in test_loader:
                        input_ids = batch["input_ids"].to(device)
                        attention_mask = batch["attention_mask"].to(device)
                        features_t = batch["features"].to(device)
                        
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            features=features_t if config["use_features"] else None,
                        )
                        preds = torch.argmax(outputs["logits"], dim=-1)
                        all_preds.extend(preds.cpu().numpy().tolist())
                        all_labels_test.extend(batch["labels"].numpy().tolist())
                
                test_output_dir = os.path.join(args.output_dir, "test_results")
                evaluate_and_save(all_labels_test, all_preds, test_datasets,
                                  test_output_dir,
                                  config["display_name"] + " (TEST SET)")
            else:
                print(f"  No best model found at {best_model_path}")
        
        elif args.experiment == "tfidf":
            print("  TF-IDF test evaluation not yet implemented")
        elif args.experiment == "features_only":
            print("  Features-only test evaluation not yet implemented")
    
    print(f"\n{'='*60}")
    print("DONE!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()