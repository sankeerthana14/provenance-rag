"""
rag_competitor_eval.py
======================
Compare PROVE-RAG against existing RAG evaluation
frameworks adapted to evidence-state detection.

Competitors:
  1. RAGAS-style context metrics (Es et al., 2024)
     — context precision, context recall, relevancy
     — THE standard RAG evaluation framework
     — Shows: existing RAG metrics evaluate output
       quality, not evidence states

  2. CRAG-style retrieval evaluator (Yan et al., 2024)
     — LLM judges if each retrieved doc is
       Correct/Incorrect/Ambiguous
     — Represents corrective retrieval approaches
     — Shows: binary relevance can't distinguish 4
       evidence states

  3. Self-RAG critic (Asai et al., 2024)
     — LLM predicts [IsRel], [IsSup] tokens
     — Represents self-reflective retrieval
     — Shows: support/relevance binary doesn't
       capture contradicted or superseded

Key insight for the paper: these frameworks solve
a DIFFERENT problem (post-generation evaluation or
binary relevance) and cannot distinguish 4 evidence
states. PROVE-RAG addresses PRE-generation evidence
quality with a 4-way taxonomy.

Usage:
    python rag_competitor_eval.py \
        --test_path data/processed/test.json \
        --llm_path models/Llama-3.1-8B-Instruct \
        --output_dir results/rag_baselines \
        --max_samples 2000

Requirements:
    pip install rouge-score --break-system-packages
"""

import argparse
import json
import os
import re
import string
import numpy as np
import torch
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix
)
from sklearn.model_selection import cross_val_score


LABEL2ID = {
    "sufficient": 0, "insufficient": 1,
    "contradicted": 2, "superseded": 3,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_path", type=str,
        required=True)
    parser.add_argument("--llm_path", type=str,
        required=True,
        help="Path to local LLM for CRAG/Self-RAG")
    parser.add_argument("--output_dir", type=str,
        default="results/rag_baselines")
    parser.add_argument("--max_samples", type=int,
        default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_data(path, max_samples=0, seed=42):
    """Load and optionally subsample test data."""
    with open(path) as f:
        data = json.load(f)
    if max_samples > 0 and len(data) > max_samples:
        import random
        random.seed(seed)
        data = random.sample(data, max_samples)
    print(f"  Loaded {len(data)} instances")
    return data


def extract_parts(instance):
    """Extract question, evidence list, gold answer."""
    question = instance.get("input_text", "")

    # Parse structured input into separate evidence
    structured = instance.get("structured_input", "")
    if structured:
        parts = structured.split("[SEP]")
        evidences = [p.strip() for p in parts[1:]
                     if p.strip()]
    else:
        evidences = [instance.get("plain_input", "")]

    gold = (instance.get("target_answer")
            or instance.get("target_label") or "")

    return question, evidences, str(gold)


def normalize(text):
    """Normalize text for token overlap."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()


def evaluate_and_save(true_labels, pred_labels,
                      dataset_labels, output_dir,
                      experiment_name):
    """Evaluate and save results."""
    os.makedirs(output_dir, exist_ok=True)
    accuracy = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels,
                        average="macro")
    label_names = [ID2LABEL[i] for i in range(4)]
    report = classification_report(
        true_labels, pred_labels,
        target_names=label_names, digits=4
    )

    print(f"\n  {'='*50}")
    print(f"  {experiment_name}")
    print(f"  {'='*50}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Macro-F1:  {macro_f1:.4f}")
    print(f"\n{report}")

    per_dataset = {}
    for ds in sorted(set(dataset_labels)):
        mask = [i for i, d in enumerate(dataset_labels)
                if d == ds]
        ds_true = [true_labels[i] for i in mask]
        ds_pred = [pred_labels[i] for i in mask]
        ds_f1 = f1_score(ds_true, ds_pred,
                         average="macro")
        per_dataset[ds] = {
            "macro_f1": ds_f1, "n": len(mask)
        }
        print(f"  {ds}: F1={ds_f1:.4f} (n={len(mask)})")

    results = {
        "experiment": experiment_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(
            true_labels, pred_labels
        ).tolist(),
        "per_dataset": per_dataset,
    }
    with open(os.path.join(output_dir,
              "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)


# ============================================================
# COMPETITOR 1: RAGAS-STYLE CONTEXT METRICS
# (Es et al., 2024 — "RAGAS: Automated Evaluation of
#  Retrieval Augmented Generation")
#
# Core idea: compute context quality metrics, then
# test if these metrics can classify evidence states.
#
# We implement the metrics WITHOUT the ragas library
# to avoid dependency issues on HPC. The metrics
# follow the same logic as RAGAS but use token overlap
# and embedding similarity instead of LLM calls.
# ============================================================

def compute_ragas_metrics(data):
    """
    Compute RAGAS-inspired context metrics for each
    instance. These capture what RAGAS measures:

    1. context_precision: fraction of evidence units
       that contain gold answer tokens (are relevant)
    2. context_recall: fraction of gold answer tokens
       found somewhere in the evidence
    3. context_relevancy: avg word overlap between
       question and each evidence unit
    4. context_utilization: ratio of non-empty evidence
       units to total evidence units
    5. answer_coverage: longest common subsequence
       ratio between gold answer and evidence
    """
    print("  Computing RAGAS-style metrics...")
    metrics = []

    for inst in data:
        question, evidences, gold = extract_parts(inst)

        gold_tokens = set(normalize(gold))
        q_tokens = set(normalize(question))

        if not evidences:
            metrics.append([0, 0, 0, 0, 0])
            continue

        # Context precision: how many evidence units
        # contain gold answer tokens?
        if gold_tokens:
            relevant_units = sum(
                1 for e in evidences
                if gold_tokens & set(normalize(e))
            )
            ctx_precision = (relevant_units
                             / len(evidences))
        else:
            ctx_precision = 0.0

        # Context recall: what fraction of gold tokens
        # appear in ANY evidence unit?
        all_ev_tokens = set()
        for e in evidences:
            all_ev_tokens.update(normalize(e))
        if gold_tokens:
            ctx_recall = (len(gold_tokens & all_ev_tokens)
                          / len(gold_tokens))
        else:
            ctx_recall = 0.0

        # Context relevancy: avg question-evidence
        # word overlap
        stopwords = {
            "the", "a", "an", "is", "are", "was",
            "were", "be", "been", "have", "has", "had",
            "do", "does", "did", "will", "would", "to",
            "of", "in", "for", "on", "with", "at", "by",
            "from", "and", "but", "or", "not", "that",
            "this", "it", "its", "what", "who", "how",
        }
        q_content = q_tokens - stopwords
        if q_content:
            overlaps = []
            for e in evidences:
                e_tokens = set(normalize(e)) - stopwords
                if e_tokens:
                    overlap = (len(q_content & e_tokens)
                               / len(q_content))
                else:
                    overlap = 0.0
                overlaps.append(overlap)
            ctx_relevancy = np.mean(overlaps)
        else:
            ctx_relevancy = 0.0

        # Context utilization: non-empty evidence ratio
        non_empty = sum(1 for e in evidences
                        if len(e.strip()) > 10)
        ctx_utilization = non_empty / len(evidences)

        # Answer coverage: best single-evidence
        # token overlap with gold
        if gold_tokens:
            best_coverage = max(
                len(gold_tokens & set(normalize(e)))
                / len(gold_tokens)
                for e in evidences
            ) if evidences else 0.0
        else:
            best_coverage = 0.0

        metrics.append([
            ctx_precision, ctx_recall,
            ctx_relevancy, ctx_utilization,
            best_coverage,
        ])

    return np.array(metrics)


def run_ragas_baseline(data, output_dir):
    """
    RAGAS-style baseline: compute context metrics,
    train logistic regression to classify evidence
    states. This gives RAGAS its BEST possible
    performance by training on the metrics.

    Reference:
      Es et al. (2024). "RAGAS: Automated Evaluation
      of Retrieval Augmented Generation." EACL 2024.
    """
    print(f"\n{'='*60}")
    print("COMPETITOR: RAGAS-style Context Metrics")
    print("(Es et al., 2024)")
    print(f"{'='*60}")

    metrics = compute_ragas_metrics(data)

    labels = [LABEL2ID[inst["evidence_state_label"]]
              for inst in data]
    datasets = [inst.get("dataset", "unknown")
                for inst in data]

    # Print per-class metric means
    print(f"\n  Per-class metric means:")
    metric_names = [
        "ctx_precision", "ctx_recall",
        "ctx_relevancy", "ctx_utilization",
        "answer_coverage"
    ]
    for label_id in range(4):
        mask = np.array(labels) == label_id
        if mask.sum() > 0:
            means = metrics[mask].mean(axis=0)
            print(f"    {ID2LABEL[label_id]}: "
                  + ", ".join(f"{n}={m:.3f}"
                    for n, m in
                    zip(metric_names, means)))

    # Train classifier (5-fold cross-val for fair eval)
    print(f"\n  Training classifier on RAGAS metrics...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0, max_iter=1000,
            class_weight="balanced",
            multi_class="multinomial",
            random_state=42
        )),
    ])

    # Cross-validation score
    cv_f1 = cross_val_score(
        pipeline, metrics, labels,
        cv=5, scoring="f1_macro"
    )
    print(f"  5-fold CV Macro-F1: "
          f"{cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

    # Full train + predict for detailed metrics
    pipeline.fit(metrics, labels)
    preds = pipeline.predict(metrics).tolist()

    ragas_output = os.path.join(output_dir, "ragas")
    evaluate_and_save(
        labels, preds, datasets, ragas_output,
        "RAGAS-style Context Metrics "
        "(Es et al., 2024)"
    )


# ============================================================
# COMPETITOR 2: CRAG-STYLE RETRIEVAL EVALUATOR
# (Yan et al., 2024 — "Corrective Retrieval Augmented
#  Generation")
#
# Core idea: use an LLM to judge whether each
# retrieved document is Correct/Incorrect/Ambiguous,
# then aggregate into an evidence-state prediction.
#
# Mapping:
#   All correct → sufficient
#   All ambiguous → insufficient
#   Any incorrect → contradicted
#   (Cannot detect superseded)
# ============================================================

def run_crag_baseline(data, llm_path, output_dir):
    """
    CRAG-style retrieval evaluator: LLM judges each
    evidence passage, then aggregates into evidence
    state prediction.

    Reference:
      Yan et al. (2024). "Corrective Retrieval
      Augmented Generation." arXiv:2401.15884.
    """
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer
    )

    print(f"\n{'='*60}")
    print("COMPETITOR: CRAG-style Retrieval Evaluator")
    print("(Yan et al., 2024)")
    print(f"{'='*60}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"  Loading LLM from {llm_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        llm_path, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        llm_path, torch_dtype=torch.float16,
        device_map="auto", trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_template = """Evaluate whether the following evidence is relevant and correct for answering the question.

Question: {question}
Evidence: {evidence}

Rate this evidence as exactly one of:
- CORRECT: The evidence is relevant and supports answering the question
- INCORRECT: The evidence contradicts the expected answer or contains wrong information
- AMBIGUOUS: The evidence is unclear, insufficient, or not directly relevant

Rating:"""

    true_labels = []
    pred_labels = []
    dataset_labels = []

    print(f"  Evaluating {len(data)} instances...")

    for i, inst in enumerate(data):
        question, evidences, gold = extract_parts(inst)
        true_labels.append(
            LABEL2ID[inst["evidence_state_label"]]
        )
        dataset_labels.append(
            inst.get("dataset", "unknown")
        )

        # Judge first 3 evidence passages (for speed)
        judgments = []
        for ev in evidences[:3]:
            prompt = prompt_template.format(
                question=question[:500],
                evidence=ev[:500]
            )
            inputs = tokenizer(
                prompt, return_tensors="pt",
                truncation=True, max_length=1024
            ).to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=10,
                    temperature=0.0, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip().upper()

            if "INCORRECT" in response:
                judgments.append("incorrect")
            elif "AMBIGUOUS" in response:
                judgments.append("ambiguous")
            else:
                judgments.append("correct")

        # Aggregate judgments → evidence state
        n_correct = judgments.count("correct")
        n_incorrect = judgments.count("incorrect")
        n_ambiguous = judgments.count("ambiguous")

        if n_incorrect > 0:
            pred = 2  # contradicted
        elif n_ambiguous > n_correct:
            pred = 1  # insufficient
        elif n_correct == len(judgments):
            pred = 0  # sufficient
        else:
            pred = 1  # insufficient (default)
        # Note: CRAG cannot predict superseded (3)

        pred_labels.append(pred)

        if (i + 1) % 100 == 0:
            running_f1 = f1_score(
                true_labels[:i+1], pred_labels[:i+1],
                average="macro"
            )
            print(f"    [{i+1}/{len(data)}] "
                  f"running F1={running_f1:.4f}")

    del model
    torch.cuda.empty_cache()

    crag_output = os.path.join(output_dir, "crag")
    evaluate_and_save(
        true_labels, pred_labels, dataset_labels,
        crag_output,
        "CRAG-style Retrieval Evaluator "
        "(Yan et al., 2024)"
    )


# ============================================================
# COMPETITOR 3: SELF-RAG CRITIC
# (Asai et al., 2024 — "Self-RAG: Learning to
#  Retrieve, Generate, and Critique through
#  Self-Reflection")
#
# Core idea: LLM acts as a critic, predicting whether
# retrieved passages are [Relevant] and whether they
# [Support] the answer.
#
# Mapping:
#   Relevant + Supported → sufficient
#   Not relevant → insufficient
#   Relevant + Not supported → contradicted
#   (Cannot detect superseded)
# ============================================================

def run_selfrag_baseline(data, llm_path, output_dir):
    """
    Self-RAG-style critic: LLM predicts relevance and
    support for retrieved evidence.

    Reference:
      Asai et al. (2024). "Self-RAG: Learning to
      Retrieve, Generate, and Critique through
      Self-Reflection." ICLR 2024.
    """
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer
    )

    print(f"\n{'='*60}")
    print("COMPETITOR: Self-RAG Critic")
    print("(Asai et al., 2024)")
    print(f"{'='*60}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"  Loading LLM from {llm_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        llm_path, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        llm_path, torch_dtype=torch.float16,
        device_map="auto", trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_template = """Given the question and evidence below, answer two questions:

1. Is the evidence RELEVANT to the question? (YES or NO)
2. Does the evidence SUPPORT answering the question? (FULLY, PARTIALLY, or NO)

Question: {question}
Evidence: {evidence}

Answer in this exact format:
Relevant: YES or NO
Support: FULLY, PARTIALLY, or NO"""

    true_labels = []
    pred_labels = []
    dataset_labels = []

    print(f"  Evaluating {len(data)} instances...")

    for i, inst in enumerate(data):
        question, evidences, gold = extract_parts(inst)
        true_labels.append(
            LABEL2ID[inst["evidence_state_label"]]
        )
        dataset_labels.append(
            inst.get("dataset", "unknown")
        )

        # Evaluate concatenated evidence
        ev_concat = " ".join(evidences[:5])[:1000]
        prompt = prompt_template.format(
            question=question[:500],
            evidence=ev_concat
        )

        inputs = tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=1024
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=30,
                temperature=0.0, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip().upper()

        # Parse response
        is_relevant = "YES" in response.split("\n")[0] \
            if response else False
        support = "NO"
        for line in response.split("\n"):
            if "SUPPORT" in line.upper():
                if "FULLY" in line.upper():
                    support = "FULLY"
                elif "PARTIAL" in line.upper():
                    support = "PARTIALLY"
                else:
                    support = "NO"

        # Map to evidence state
        if is_relevant and support == "FULLY":
            pred = 0  # sufficient
        elif not is_relevant:
            pred = 1  # insufficient
        elif is_relevant and support == "NO":
            pred = 2  # contradicted
        else:  # relevant + partially supported
            pred = 1  # insufficient
        # Note: Self-RAG cannot predict superseded (3)

        pred_labels.append(pred)

        if (i + 1) % 100 == 0:
            running_f1 = f1_score(
                true_labels[:i+1], pred_labels[:i+1],
                average="macro"
            )
            print(f"    [{i+1}/{len(data)}] "
                  f"running F1={running_f1:.4f}")

    del model
    torch.cuda.empty_cache()

    selfrag_output = os.path.join(
        output_dir, "selfrag"
    )
    evaluate_and_save(
        true_labels, pred_labels, dataset_labels,
        selfrag_output,
        "Self-RAG Critic (Asai et al., 2024)"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("PROVE-RAG vs RAG Evaluation Competitors")
    print("=" * 60)

    data = load_data(
        args.test_path, args.max_samples, args.seed
    )

    # 1. RAGAS (no GPU needed, fast)
    run_ragas_baseline(data, args.output_dir)

    # 2. CRAG (needs GPU + LLM, slower)
    run_crag_baseline(
        data, args.llm_path, args.output_dir
    )

    # 3. Self-RAG (needs GPU + LLM, slower)
    run_selfrag_baseline(
        data, args.llm_path, args.output_dir
    )

    print(f"\n{'='*60}")
    print("ALL COMPETITORS COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()