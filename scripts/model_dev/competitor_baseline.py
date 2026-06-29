"""
competitor_baselines.py
=======================
Competitor baselines for evidence-state detection. Implements three paradigms that compares PROVE-RAG is compared against:

1. NLI-based (roberta-large-mnli)
   — Represents the NLP entailment approach.
   — Maps entailment→sufficient, contradiction→contradicted,
     neutral→insufficient. Cannot detect superseded.
   — Landmark: Thorne et al. (FEVER), Nie et al. (2020)

2. Embedding Similarity (cosine sim thresholding)
   — Represents what RAG systems actually use for
     evidence scoring (dense retrieval relevance).
   — Shows that relevance ≠ evidence quality.
   — Landmark: Karpukhin et al. (DPR, 2020)

3. LLM zero-shot
   — Already implemented in train_detector.py
   — Run separately via the SLURM command provided.

Usage:
    python competitor_baselines.py \
        --test_path data/processed/test.json \
        --nli_model_path models/roberta-large-mnli \
        --output_dir results/baselines \
        --max_samples 0

    (--max_samples 0 means use all test data)
"""

import argparse
import json
import os
import numpy as np
import torch
from collections import Counter
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix
)


LABEL2ID = {
    "sufficient": 0, "insufficient": 1,
    "contradicted": 2, "superseded": 3,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = 4
FEATURE_NAMES = [
    "source_diversity", "text_resolution_rate",
    "avg_evidence_length", "min_evidence_length",
    "duplicate_rate", "document_overlap_rate",
    "entity_overlap", "evidence_count",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Competitor baselines for "
        "evidence-state detection"
    )
    parser.add_argument("--test_path", type=str,
        required=True,
        help="Path to test.json")
    parser.add_argument("--nli_model_path", type=str,
        default="roberta-large-mnli",
        help="Path to NLI model")
    parser.add_argument("--output_dir", type=str,
        default="results/baselines")
    parser.add_argument("--max_samples", type=int,
        default=0,
        help="Max samples (0 = all)")
    parser.add_argument("--batch_size", type=int,
        default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_test_data(path, max_samples=0):
    """Load test data."""
    print(f"  Loading {path}...")
    with open(path, "r") as f:
        data = json.load(f)
    if max_samples > 0:
        import random
        random.seed(42)
        data = random.sample(data, min(max_samples,
                                       len(data)))
    print(f"  Loaded {len(data)} instances")
    return data


def extract_question_and_evidence(instance):
    """
    Extract question and evidence text separately.
    For NLI: question=premise, evidence=hypothesis.
    """
    question = instance.get("input_text", "")

    structured = instance.get("structured_input", "")
    if structured:
        parts = structured.split("[SEP]")
        # First part is question, rest is evidence
        evidence_parts = [p.strip() for p in parts[1:]
                         if p.strip()]
        evidence = " ".join(evidence_parts)
    else:
        evidence = instance.get("plain_input", "")
        # Remove question from plain input
        if evidence.startswith(question):
            evidence = evidence[len(question):].strip()

    return question, evidence


def evaluate_predictions(true_labels, pred_labels,
                        dataset_labels, output_dir,
                        experiment_name):
    """Compute all metrics and save results."""
    os.makedirs(output_dir, exist_ok=True)

    accuracy = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels,
                        average="macro")

    label_names = [ID2LABEL[i] for i in range(NUM_LABELS)]
    report = classification_report(
        true_labels, pred_labels,
        target_names=label_names, digits=4
    )
    cm = confusion_matrix(true_labels, pred_labels)

    print(f"\n  {'='*50}")
    print(f"  RESULTS: {experiment_name}")
    print(f"  {'='*50}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Macro-F1:  {macro_f1:.4f}")
    print(f"\n{report}")

    # Per-dataset breakdown
    per_dataset = {}
    if dataset_labels:
        for ds in sorted(set(dataset_labels)):
            mask = [i for i, d in enumerate(dataset_labels)
                    if d == ds]
            ds_true = [true_labels[i] for i in mask]
            ds_pred = [pred_labels[i] for i in mask]
            ds_f1 = f1_score(ds_true, ds_pred,
                            average="macro")
            ds_acc = accuracy_score(ds_true, ds_pred)
            per_dataset[ds] = {
                "accuracy": ds_acc,
                "macro_f1": ds_f1,
                "n": len(mask),
            }
            print(f"  {ds}: F1={ds_f1:.4f}, "
                  f"Acc={ds_acc:.4f} (n={len(mask)})")

    results = {
        "experiment": experiment_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "per_dataset": per_dataset,
    }

    out_path = os.path.join(output_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved to {out_path}")

    return results


# ============================================================
# BASELINE 1: NLI (roberta-large-mnli)
# ============================================================

def run_nli_baseline(test_data, nli_model_path,
                     output_dir, batch_size=32):
    """
    NLI-based evidence-state detection.

    Uses a pre-trained NLI model to classify the
    relationship between question (premise) and
    evidence (hypothesis).

    Mapping:
      entailment   → sufficient (evidence supports)
      contradiction → contradicted (evidence conflicts)
      neutral       → insufficient (evidence doesn't
                      help; also catches superseded
                      since NLI has no temporal class)

    This baseline represents the standard NLP approach
    to fact verification (FEVER-style). It cannot
    distinguish insufficient from superseded, which is
    WHY provenance features are needed.

    Landmark references:
      - Thorne et al. (2018) FEVER
      - Nie et al. (2020) Adversarial NLI
      - Williams et al. (2018) MultiNLI
    """
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification
    )

    print(f"\n{'='*60}")
    print("BASELINE: NLI (roberta-large-mnli)")
    print(f"Model: {nli_model_path}")
    print(f"{'='*60}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load model
    print(f"  Loading NLI model...")
    tokenizer = AutoTokenizer.from_pretrained(
        nli_model_path
    )
    model = AutoModelForSequenceClassification \
        .from_pretrained(nli_model_path).to(device)
    model.eval()

    # NLI label mapping
    # roberta-large-mnli outputs:
    #   0=contradiction, 1=neutral, 2=entailment
    NLI_TO_EVIDENCE = {
        2: 0,  # entailment → sufficient
        1: 1,  # neutral → insufficient
        0: 2,  # contradiction → contradicted
        # superseded (3) is never predicted by NLI
    }

    # Process in batches
    true_labels = []
    pred_labels = []
    dataset_labels = []

    print(f"  Running inference on {len(test_data)} "
          f"instances...")

    for i in range(0, len(test_data), batch_size):
        batch_data = test_data[i:i + batch_size]

        premises = []
        hypotheses = []

        for inst in batch_data:
            q, e = extract_question_and_evidence(inst)
            # Truncate evidence for NLI
            premises.append(q[:512])
            hypotheses.append(e[:512])
            true_labels.append(
                LABEL2ID[inst["evidence_state_label"]]
            )
            dataset_labels.append(
                inst.get("dataset", "unknown")
            )

        # Tokenize
        inputs = tokenizer(
            premises, hypotheses,
            truncation=True, padding=True,
            max_length=512, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            nli_preds = torch.argmax(
                outputs.logits, dim=-1
            ).cpu().tolist()

        # Map NLI predictions to evidence states
        for nli_pred in nli_preds:
            evidence_pred = NLI_TO_EVIDENCE.get(
                nli_pred, 1  # default to insufficient
            )
            pred_labels.append(evidence_pred)

        if (i + batch_size) % 1000 < batch_size:
            print(f"    Processed {min(i + batch_size, len(test_data))}"
                  f"/{len(test_data)}")

    # Clean up
    del model
    torch.cuda.empty_cache()

    # Evaluate
    nli_output = os.path.join(output_dir, "nli")
    evaluate_predictions(
        true_labels, pred_labels, dataset_labels,
        nli_output,
        "NLI Baseline (roberta-large-mnli)"
    )


# ============================================================
# BASELINE 2: EMBEDDING SIMILARITY
# ============================================================

def run_embedding_similarity(test_data, nli_model_path,
                             output_dir, batch_size=32):
    """
    Embedding similarity baseline for evidence-state
    detection.

    Computes cosine similarity between question and
    evidence embeddings, then uses learned thresholds
    to classify evidence states.

    This represents what RAG systems actually use:
    dense retrieval scoring based on semantic
    similarity. The key insight this baseline
    demonstrates is that RELEVANCE ≠ EVIDENCE QUALITY.
    Evidence can be highly relevant (similar embedding)
    but contradicted or superseded.

    Uses the same RoBERTa-large backbone as NLI for
    fair comparison (just the encoder, no NLI head).

    Landmark references:
      - Karpukhin et al. (2020) DPR
      - Reimers & Gurevych (2019) Sentence-BERT
    """
    from transformers import AutoTokenizer, AutoModel

    print(f"\n{'='*60}")
    print("BASELINE: Embedding Similarity")
    print(f"{'='*60}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load encoder (use RoBERTa-large from NLI model)
    print(f"  Loading encoder from {nli_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        nli_model_path
    )
    model = AutoModel.from_pretrained(
        nli_model_path
    ).to(device)
    model.eval()

    def get_embedding(texts, max_length=256):
        """Get CLS embeddings for a batch of texts."""
        inputs = tokenizer(
            texts, truncation=True, padding=True,
            max_length=max_length, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # CLS token
        return outputs.last_hidden_state[:, 0, :].cpu()

    # Compute similarities
    print(f"  Computing similarities for "
          f"{len(test_data)} instances...")
    similarities = []
    true_labels = []
    dataset_labels = []

    for i in range(0, len(test_data), batch_size):
        batch_data = test_data[i:i + batch_size]

        questions = []
        evidences = []

        for inst in batch_data:
            q, e = extract_question_and_evidence(inst)
            questions.append(q[:256])
            evidences.append(e[:256])
            true_labels.append(
                LABEL2ID[inst["evidence_state_label"]]
            )
            dataset_labels.append(
                inst.get("dataset", "unknown")
            )

        # Get embeddings
        q_emb = get_embedding(questions)
        e_emb = get_embedding(evidences)

        # Cosine similarity
        q_norm = q_emb / (
            q_emb.norm(dim=1, keepdim=True) + 1e-8
        )
        e_norm = e_emb / (
            e_emb.norm(dim=1, keepdim=True) + 1e-8
        )
        cos_sim = (q_norm * e_norm).sum(dim=1).tolist()
        similarities.extend(cos_sim)

        if (i + batch_size) % 1000 < batch_size:
            print(f"    Processed "
                  f"{min(i + batch_size, len(test_data))}"
                  f"/{len(test_data)}")

    # Clean up
    del model
    torch.cuda.empty_cache()

    similarities = np.array(similarities)
    true_labels_np = np.array(true_labels)

    # Learn optimal thresholds from the data
    # (This gives the baseline its BEST possible
    #  performance — fair to the competitor)
    print(f"\n  Similarity stats:")
    for label_id in range(NUM_LABELS):
        mask = true_labels_np == label_id
        if mask.sum() > 0:
            sims = similarities[mask]
            print(f"    {ID2LABEL[label_id]}: "
                  f"mean={sims.mean():.4f}, "
                  f"std={sims.std():.4f}, "
                  f"median={np.median(sims):.4f}")

    # Strategy 1: Optimal single threshold
    # (best binary: high sim → sufficient, low → rest)
    best_f1 = 0
    best_threshold = 0.5
    for t in np.arange(0.0, 1.0, 0.01):
        preds = np.where(
            similarities >= t, 0, 1
        )  # sufficient or insufficient
        f1 = f1_score(true_labels, preds,
                      average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    # Strategy 2: Multi-threshold
    # Sort similarities and find thresholds that
    # separate the 4 classes as well as possible
    # Use percentile-based thresholds
    percentiles = np.percentile(
        similarities, [25, 50, 75]
    )
    t_low, t_mid, t_high = percentiles

    pred_labels = []
    for sim in similarities:
        if sim >= t_high:
            pred_labels.append(0)  # sufficient
        elif sim >= t_mid:
            pred_labels.append(1)  # insufficient
        elif sim >= t_low:
            pred_labels.append(2)  # contradicted
        else:
            pred_labels.append(3)  # superseded

    multi_f1 = f1_score(true_labels, pred_labels,
                        average="macro")

    # Use whichever strategy is better
    if best_f1 > multi_f1:
        print(f"\n  Using binary threshold: {best_threshold:.2f}")
        pred_labels = [
            0 if s >= best_threshold else 1
            for s in similarities
        ]
        strategy = "binary_threshold"
    else:
        print(f"\n  Using multi-threshold: "
              f"{t_low:.3f}, {t_mid:.3f}, {t_high:.3f}")
        strategy = "multi_threshold"

    print(f"  Strategy: {strategy}")

    # Evaluate
    sim_output = os.path.join(
        output_dir, "embedding_similarity"
    )
    evaluate_predictions(
        true_labels, pred_labels, dataset_labels,
        sim_output,
        f"Embedding Similarity ({strategy})"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    print("=" * 60)
    print("PROVE-RAG Competitor Baselines")
    print("=" * 60)

    # Load test data
    test_data = load_test_data(
        args.test_path, args.max_samples
    )

    # Run NLI baseline
    run_nli_baseline(
        test_data, args.nli_model_path,
        args.output_dir, args.batch_size
    )

    # Run embedding similarity baseline
    run_embedding_similarity(
        test_data, args.nli_model_path,
        args.output_dir, args.batch_size
    )

    print(f"\n{'='*60}")
    print("DONE! All baselines complete.")
    print(f"{'='*60}")
    print(f"\nReminder: Run LLM zero-shot separately:")
    print(f"  python train_detector.py "
          f"--experiment llm_zeroshot ...")


if __name__ == "__main__":
    main()
