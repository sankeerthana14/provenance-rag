"""
end_to_end_eval.py
==================
End-to-end evaluation of PROVE-RAG's evidence-state detector
in an agentic RAG pipeline.

Compares three conditions:
  1. Naive RAG:    Always answers from whatever evidence is provided
  2. PROVE-RAG:    Detects evidence state → corrective action → then answers
  3. Oracle RAG:   Uses ground-truth evidence states (upper bound)

Actions on non-sufficient evidence:
  - insufficient  → retrieve more evidence (simulated: swap to sufficient variant)
  - contradicted  → cross-check & retrieve (simulated: swap to sufficient variant)
  - superseded    → retrieve updated evidence (simulated: swap to sufficient variant)
  - sufficient    → answer directly (no swap)

Usage:
    python end_to_end_eval.py \
        --test_path data/processed/test.json \
        --detector_dir results/roberta_feat \
        --detector_base_model models/roberta-base \
        --llm_path models/Llama-3.1-8B-Instruct \
        --output_dir results/end_to_end \
        --max_questions 500
"""

import argparse
import json
import os
import re
import string
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple


# ============================================================
# CONFIG (must match train_detector.py)
# ============================================================

LABEL2ID = {"sufficient": 0, "insufficient": 1, "contradicted": 2, "superseded": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = 4
FEATURE_NAMES = [
    "source_diversity", "text_resolution_rate", "avg_evidence_length",
    "min_evidence_length", "duplicate_rate", "document_overlap_rate",
    "entity_overlap", "evidence_count",
]
NUM_FEATURES = len(FEATURE_NAMES)


# ============================================================
# DETECTOR MODEL (copied from train_detector.py for standalone use)
# ============================================================

class DebertaWithFeatures(nn.Module):
    def __init__(self, model_name, num_labels, num_features, use_features=False, dropout=0.1):
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
        cls_output = outputs.last_hidden_state[:, 0, :].float()
        cls_output = self.dropout(cls_output)
        if self.use_features and features is not None:
            cls_output = torch.cat([cls_output, features.float()], dim=-1)
        logits = self.classifier(cls_output)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}


# ============================================================
# ANSWER EVALUATION METRICS
# ============================================================

def normalize_answer(s: str) -> str:
    """Normalize answer for evaluation (lowercase, remove articles/punctuation)."""
    s = s.lower().strip()
    # Remove articles
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    # Remove punctuation
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    # Collapse whitespace
    s = ' '.join(s.split())
    return s


def compute_f1(prediction: str, gold: str) -> float:
    """Token-level F1 between prediction and gold answer."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return 0.0

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_em(prediction: str, gold: str) -> float:
    """Exact match (after normalization)."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


# ============================================================
# DATA LOADING
# ============================================================

def load_and_group_data(test_path: str, max_questions: Optional[int] = None, seed: int = 42):
    """
    Load test data and group by question.
    Returns evaluation examples with sufficient evidence lookup.
    """
    print(f"  Loading {test_path}...")
    with open(test_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} instances")

    # Check for gold answers
    has_gold = any(inst.get("target_answer") or inst.get("target_label")
                   or inst.get("answer") for inst in data[:100])
    if not has_gold:
        print("\n  WARNING: No gold answers found in data!")
        print("  Fields found:", list(data[0].keys()) if data else "none")
        print("  The script will generate reference answers from sufficient evidence.")
        print("  For better evaluation, add 'answer' field to your preprocessed data.\n")

    # Group by question (input_text)
    question_groups = defaultdict(list)
    for inst in data:
        question_groups[inst["input_text"]].append(inst)

    print(f"  Found {len(question_groups)} unique questions")

    # Check how many questions have all 4 variants
    complete = sum(1 for q, variants in question_groups.items()
                   if len(set(v["evidence_state_label"] for v in variants)) == 4)
    print(f"  Questions with all 4 variants: {complete}")

    # Build evaluation set
    eval_questions = []
    for question_text, variants in question_groups.items():
        # Find sufficient variant
        sufficient = None
        others = []
        for v in variants:
            if v["evidence_state_label"] == "sufficient":
                sufficient = v
            else:
                others.append(v)

        if sufficient is None:
            continue  # Skip questions without sufficient variant

        # Get gold answer (target_answer for QA, target_label for FEVER)
        gold_answer = (sufficient.get("target_answer")
                       or sufficient.get("target_label")
                       or sufficient.get("answer")
                       or None)

        eval_questions.append({
            "question_text": question_text,
            "sufficient_variant": sufficient,
            "other_variants": others,
            "gold_answer": gold_answer,
            "dataset": sufficient.get("dataset", "unknown"),
        })

    print(f"  Evaluation questions (with sufficient variant): {len(eval_questions)}")

    # Subsample if needed
    if max_questions and len(eval_questions) > max_questions:
        import random
        random.seed(seed)
        eval_questions = random.sample(eval_questions, max_questions)
        print(f"  Subsampled to {max_questions} questions")

    return eval_questions, has_gold


def extract_evidence_for_llm(instance: dict) -> str:
    """Extract readable evidence text from a data instance."""
    # Use structured_input and reformat for LLM
    structured = instance.get("structured_input", "")
    if structured:
        # Parse "Question: ... [SEP] Evidence 1: ... [SEP] Evidence 2: ..."
        parts = structured.split("[SEP]")
        formatted_parts = []
        for p in parts:
            p = p.strip()
            if p:
                formatted_parts.append(p)
        return "\n".join(formatted_parts)

    # Fallback: use evidence_text or plain_input
    if instance.get("evidence_text"):
        return f"Question: {instance['input_text']}\n{instance['evidence_text']}"

    return instance.get("plain_input", instance.get("input_text", ""))


def get_features(instance: dict) -> List[float]:
    """Extract graph features from instance."""
    gf = instance.get("graph_features", {})
    return [float(gf.get(fn, 0.0)) for fn in FEATURE_NAMES]


# ============================================================
# DETECTOR
# ============================================================

class EvidenceStateDetector:
    """Wrapper for the trained evidence-state detector."""

    def __init__(self, detector_dir: str, base_model: str, device: torch.device):
        from transformers import AutoTokenizer

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)

        # Load model
        self.model = DebertaWithFeatures(
            model_name=base_model,
            num_labels=NUM_LABELS,
            num_features=NUM_FEATURES,
            use_features=True,  # Approach B
        ).to(device).float()

        model_path = os.path.join(detector_dir, "best_model.pt")
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

        # Load feature normalization
        norm_path = os.path.join(detector_dir, "feature_norm.npz")
        if os.path.exists(norm_path):
            norm = np.load(norm_path)
            self.feat_mean = norm["mean"]
            self.feat_std = norm["std"]
        else:
            print("  WARNING: No feature normalization found, using raw features")
            self.feat_mean = np.zeros(NUM_FEATURES)
            self.feat_std = np.ones(NUM_FEATURES)

    def predict(self, instance: dict) -> str:
        """Predict evidence state for a single instance."""
        # Get text input
        text = instance.get("structured_input") or instance.get("plain_input", "")
        text = text[:5000]  # Pre-truncate

        # Tokenize
        encoding = self.tokenizer(
            text, truncation=True, padding="max_length",
            max_length=512, return_tensors="pt"
        )

        # Get features
        feat_vec = np.array(get_features(instance), dtype=np.float32)
        feat_vec = (feat_vec - self.feat_mean) / (self.feat_std + 1e-8)
        features = torch.tensor(feat_vec, dtype=torch.float32).unsqueeze(0)

        # Predict
        with torch.no_grad():
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            features = features.to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                features=features,
            )
            pred = torch.argmax(outputs["logits"], dim=-1).item()

        return ID2LABEL[pred]


# ============================================================
# LLM ANSWER GENERATOR
# ============================================================

class LLMGenerator:
    """Wrapper for Llama-3.1-8B-Instruct answer generation."""

    def __init__(self, llm_path: str, device: torch.device):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"  Loading LLM from {llm_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = device

    def generate_answer(self, evidence_text: str, dataset: str = "hotpotqa") -> str:
        """Generate answer given formatted evidence text."""
        if dataset == "fever":
            prompt = self._fever_prompt(evidence_text)
        else:
            prompt = self._qa_prompt(evidence_text)

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        # Extract just the first line/sentence as the answer
        answer = generated.split("\n")[0].strip()
        # Remove common prefixes
        for prefix in ["Answer:", "The answer is", "A:"]:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()

        return answer

    def _qa_prompt(self, evidence_text: str) -> str:
        return f"""Based on the following evidence, answer the question concisely in a few words.

{evidence_text}

Answer concisely:"""

    def _fever_prompt(self, evidence_text: str) -> str:
        return f"""Based on the following evidence, determine if the claim is SUPPORTS, REFUTES, or NOT ENOUGH INFO.

{evidence_text}

Verdict (SUPPORTS, REFUTES, or NOT ENOUGH INFO):"""

    def generate_reference(self, evidence_text: str, dataset: str = "hotpotqa") -> str:
        """Generate a reference answer from sufficient evidence (fallback when no gold answer)."""
        return self.generate_answer(evidence_text, dataset)


# ============================================================
# MAIN EVALUATION LOOP
# ============================================================

def run_evaluation(eval_questions, detector, llm, has_gold, output_dir, seed=42):
    """Run the three-condition evaluation."""
    os.makedirs(output_dir, exist_ok=True)

    results = []
    naive_scores = {"f1": [], "em": []}
    proverag_scores = {"f1": [], "em": []}
    oracle_scores = {"f1": [], "em": []}

    # Per evidence-state tracking
    state_scores = {
        state: {"naive_f1": [], "proverag_f1": [], "oracle_f1": [],
                "naive_em": [], "proverag_em": [], "oracle_em": []}
        for state in ["sufficient", "insufficient", "contradicted", "superseded"]
    }

    # Per dataset tracking
    dataset_scores = defaultdict(lambda: {
        "naive_f1": [], "proverag_f1": [], "oracle_f1": []
    })

    total_examples = sum(1 + len(q["other_variants"]) for q in eval_questions)
    print(f"\n  Running evaluation on {total_examples} examples "
          f"({len(eval_questions)} questions × ~4 variants)...")
    print(f"  Conditions: Naive RAG | PROVE-RAG | Oracle RAG\n")

    example_idx = 0

    for q_idx, q in enumerate(eval_questions):
        sufficient_variant = q["sufficient_variant"]
        sufficient_evidence = extract_evidence_for_llm(sufficient_variant)
        gold_answer = q["gold_answer"]
        dataset = q["dataset"]

        # If no gold answer, generate reference from sufficient evidence
        if gold_answer is None:
            gold_answer = llm.generate_reference(sufficient_evidence, dataset)

        # Evaluate ALL variants (including sufficient)
        all_variants = [sufficient_variant] + q["other_variants"]

        for variant in all_variants:
            true_state = variant["evidence_state_label"]
            variant_evidence = extract_evidence_for_llm(variant)

            # --- Condition 1: Naive RAG (always answer from given evidence) ---
            naive_answer = llm.generate_answer(variant_evidence, dataset)

            # --- Condition 2: PROVE-RAG (detect → act → answer) ---
            predicted_state = detector.predict(variant)
            if predicted_state == "sufficient":
                proverag_answer = llm.generate_answer(variant_evidence, dataset)
            else:
                # Remediate: use sufficient evidence instead
                proverag_answer = llm.generate_answer(sufficient_evidence, dataset)

            # --- Condition 3: Oracle RAG (use ground truth state) ---
            if true_state == "sufficient":
                oracle_answer = llm.generate_answer(variant_evidence, dataset)
            else:
                oracle_answer = llm.generate_answer(sufficient_evidence, dataset)

            # --- Evaluate ---
            naive_f1 = compute_f1(naive_answer, gold_answer)
            naive_em = compute_em(naive_answer, gold_answer)
            proverag_f1 = compute_f1(proverag_answer, gold_answer)
            proverag_em = compute_em(proverag_answer, gold_answer)
            oracle_f1 = compute_f1(oracle_answer, gold_answer)
            oracle_em = compute_em(oracle_answer, gold_answer)

            # Collect scores
            naive_scores["f1"].append(naive_f1)
            naive_scores["em"].append(naive_em)
            proverag_scores["f1"].append(proverag_f1)
            proverag_scores["em"].append(proverag_em)
            oracle_scores["f1"].append(oracle_f1)
            oracle_scores["em"].append(oracle_em)

            state_scores[true_state]["naive_f1"].append(naive_f1)
            state_scores[true_state]["proverag_f1"].append(proverag_f1)
            state_scores[true_state]["oracle_f1"].append(oracle_f1)
            state_scores[true_state]["naive_em"].append(naive_em)
            state_scores[true_state]["proverag_em"].append(proverag_em)
            state_scores[true_state]["oracle_em"].append(oracle_em)

            dataset_scores[dataset]["naive_f1"].append(naive_f1)
            dataset_scores[dataset]["proverag_f1"].append(proverag_f1)
            dataset_scores[dataset]["oracle_f1"].append(oracle_f1)

            # Track detector accuracy
            detector_correct = (predicted_state == true_state)

            results.append({
                "question": q["question_text"],
                "true_state": true_state,
                "predicted_state": predicted_state,
                "detector_correct": detector_correct,
                "dataset": dataset,
                "gold_answer": gold_answer,
                "naive_answer": naive_answer,
                "proverag_answer": proverag_answer,
                "oracle_answer": oracle_answer,
                "naive_f1": naive_f1,
                "proverag_f1": proverag_f1,
                "oracle_f1": oracle_f1,
                "naive_em": naive_em,
                "proverag_em": proverag_em,
                "oracle_em": oracle_em,
            })

            example_idx += 1
            if example_idx % 50 == 0:
                running_naive = np.mean(naive_scores["f1"])
                running_prove = np.mean(proverag_scores["f1"])
                running_oracle = np.mean(oracle_scores["f1"])
                print(f"    [{example_idx}/{total_examples}] "
                      f"Naive={running_naive:.3f}  "
                      f"PROVE-RAG={running_prove:.3f}  "
                      f"Oracle={running_oracle:.3f}")

    # ---- Print Results ----
    print(f"\n{'='*70}")
    print("END-TO-END EVALUATION RESULTS")
    print(f"{'='*70}")

    print(f"\n  OVERALL:")
    print(f"  {'Condition':<15} {'Answer F1':>10} {'Answer EM':>10}")
    print(f"  {'-'*35}")
    print(f"  {'Naive RAG':<15} {np.mean(naive_scores['f1']):>10.4f} "
          f"{np.mean(naive_scores['em']):>10.4f}")
    print(f"  {'PROVE-RAG':<15} {np.mean(proverag_scores['f1']):>10.4f} "
          f"{np.mean(proverag_scores['em']):>10.4f}")
    print(f"  {'Oracle RAG':<15} {np.mean(oracle_scores['f1']):>10.4f} "
          f"{np.mean(oracle_scores['em']):>10.4f}")

    print(f"\n  PER EVIDENCE STATE (Answer F1):")
    print(f"  {'State':<15} {'Naive':>8} {'PROVE-RAG':>10} {'Oracle':>8} {'N':>6}")
    print(f"  {'-'*50}")
    for state in ["sufficient", "insufficient", "contradicted", "superseded"]:
        ss = state_scores[state]
        n = len(ss["naive_f1"])
        if n > 0:
            print(f"  {state:<15} {np.mean(ss['naive_f1']):>8.4f} "
                  f"{np.mean(ss['proverag_f1']):>10.4f} "
                  f"{np.mean(ss['oracle_f1']):>8.4f} {n:>6}")

    print(f"\n  PER DATASET (Answer F1):")
    print(f"  {'Dataset':<15} {'Naive':>8} {'PROVE-RAG':>10} {'Oracle':>8} {'N':>6}")
    print(f"  {'-'*50}")
    for ds in sorted(dataset_scores.keys()):
        ds_s = dataset_scores[ds]
        n = len(ds_s["naive_f1"])
        if n > 0:
            print(f"  {ds:<15} {np.mean(ds_s['naive_f1']):>8.4f} "
                  f"{np.mean(ds_s['proverag_f1']):>10.4f} "
                  f"{np.mean(ds_s['oracle_f1']):>8.4f} {n:>6}")

    # Detector accuracy in this evaluation
    det_correct = sum(1 for r in results if r["detector_correct"])
    print(f"\n  Detector accuracy (in this eval): {det_correct}/{len(results)} "
          f"({det_correct/len(results):.4f})")

    # PROVE-RAG improvement over Naive
    improvement = np.mean(proverag_scores["f1"]) - np.mean(naive_scores["f1"])
    oracle_gap = np.mean(oracle_scores["f1"]) - np.mean(naive_scores["f1"])
    recovery = (improvement / oracle_gap * 100) if oracle_gap > 0 else 0
    print(f"\n  PROVE-RAG improvement over Naive: +{improvement:.4f} F1")
    print(f"  Oracle improvement over Naive:    +{oracle_gap:.4f} F1")
    print(f"  PROVE-RAG recovers {recovery:.1f}% of Oracle gap")

    # ---- Save Results ----
    summary = {
        "overall": {
            "naive_f1": np.mean(naive_scores["f1"]),
            "naive_em": np.mean(naive_scores["em"]),
            "proverag_f1": np.mean(proverag_scores["f1"]),
            "proverag_em": np.mean(proverag_scores["em"]),
            "oracle_f1": np.mean(oracle_scores["f1"]),
            "oracle_em": np.mean(oracle_scores["em"]),
            "improvement_f1": improvement,
            "oracle_gap_f1": oracle_gap,
            "recovery_pct": recovery,
        },
        "per_state": {},
        "per_dataset": {},
        "detector_accuracy": det_correct / len(results) if results else 0,
        "num_examples": len(results),
        "num_questions": len(eval_questions),
    }

    for state in ["sufficient", "insufficient", "contradicted", "superseded"]:
        ss = state_scores[state]
        if ss["naive_f1"]:
            summary["per_state"][state] = {
                "naive_f1": np.mean(ss["naive_f1"]),
                "proverag_f1": np.mean(ss["proverag_f1"]),
                "oracle_f1": np.mean(ss["oracle_f1"]),
                "naive_em": np.mean(ss["naive_em"]),
                "proverag_em": np.mean(ss["proverag_em"]),
                "oracle_em": np.mean(ss["oracle_em"]),
                "n": len(ss["naive_f1"]),
            }

    for ds in sorted(dataset_scores.keys()):
        ds_s = dataset_scores[ds]
        if ds_s["naive_f1"]:
            summary["per_dataset"][ds] = {
                "naive_f1": np.mean(ds_s["naive_f1"]),
                "proverag_f1": np.mean(ds_s["proverag_f1"]),
                "oracle_f1": np.mean(ds_s["oracle_f1"]),
                "n": len(ds_s["naive_f1"]),
            }

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(output_dir, "detailed_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved to {output_dir}/")
    return summary


# ============================================================
# MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="End-to-end evaluation of PROVE-RAG in agentic RAG pipeline"
    )
    parser.add_argument("--test_path", type=str, required=True,
                        help="Path to test.json (from preprocess_v2.py)")
    parser.add_argument("--detector_dir", type=str, required=True,
                        help="Directory with trained detector (best_model.pt + feature_norm.npz)")
    parser.add_argument("--detector_base_model", type=str, required=True,
                        help="Path to base model (e.g., models/roberta-base)")
    parser.add_argument("--llm_path", type=str, required=True,
                        help="Path to LLM (e.g., models/Llama-3.1-8B-Instruct)")
    parser.add_argument("--output_dir", type=str, default="results/end_to_end",
                        help="Output directory for results")
    parser.add_argument("--max_questions", type=int, default=500,
                        help="Max questions to evaluate (for speed)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("PROVE-RAG End-to-End Agentic RAG Evaluation")
    print("=" * 70)
    print(f"  Test data:      {args.test_path}")
    print(f"  Detector:       {args.detector_dir}")
    print(f"  Base model:     {args.detector_base_model}")
    print(f"  LLM:            {args.llm_path}")
    print(f"  Max questions:  {args.max_questions}")
    print(f"  Seed:           {args.seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device:         {device}")

    # 1. Load and group data
    print(f"\n[1/4] Loading data...")
    eval_questions, has_gold = load_and_group_data(
        args.test_path, args.max_questions, args.seed
    )

    # 2. Load detector
    print(f"\n[2/4] Loading evidence-state detector...")
    detector = EvidenceStateDetector(
        args.detector_dir, args.detector_base_model, device
    )
    print("  Detector loaded.")

    # 3. Load LLM
    print(f"\n[3/4] Loading LLM...")
    llm = LLMGenerator(args.llm_path, device)
    print("  LLM loaded.")

    # 4. Run evaluation
    print(f"\n[4/4] Running evaluation...")
    summary = run_evaluation(
        eval_questions, detector, llm, has_gold,
        args.output_dir, args.seed
    )

    print(f"\n{'='*70}")
    print("DONE!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()