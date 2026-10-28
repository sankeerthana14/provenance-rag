"""PROVE-RAG preprocessing with leakage-safe question-family splits.

This replaces the original preprocess_v2.py while keeping its CLI and model
input schema. The input files are the already-merged, variant-containing
`unified_train.json` and independently prepared `unified_val.json`.

The fundamental fix: allocate an entire normalized-question family to a
single split, rather than independently splitting (dataset, state) rows.

This is GROUP-AWARE partitioning of existing variants; it does not implement
splitting-before-variant-generation, nor does it resolve document-family,
near-duplicate, or construction-artifact problems.

Example:
    python scripts/data_preprocessing/preprocess_v2.py \
      --train_json data/modified/unified_train.json \
      --val_json data/modified/unified_val.json \
      --output_dir data/revision_2026/group_clean_v1 \
      --test_ratio 0.2 --top_k 10 --seed 42
"""

import argparse
import csv
import hashlib
import json
import os
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="PROVE-RAG group-aware preprocessing")
    parser.add_argument("--train_json", type=str, required=True)
    parser.add_argument("--val_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=10,
                        help="Maximum evidence units per example")
    parser.add_argument("--test_ratio", type=float, default=0.2,
                        help="Fraction of original training families held out for testing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_evidence_chars", type=int, default=300)
    return parser.parse_args()


def compute_word_overlap(query, text):
    """Preserved from original implementation."""
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "and", "but", "or", "not", "no", "if", "that", "this",
        "which", "who", "what", "where", "when", "how", "it", "its",
    }
    query_words = set(query.lower().split()) - stopwords
    text_words = set(text.lower().split()) - stopwords
    if not query_words or not text_words:
        return 0.0
    return len(query_words & text_words) / len(query_words)


def select_top_k_evidence(query, evidence_units, k=10, max_chars=300):
    """Preserved top-k evidence selection used in the submitted experiments."""
    scored = []
    for eu in evidence_units:
        text = eu.get("text", "")
        if not text or not text.strip():
            continue
        text = text.strip()[:max_chars]
        score = compute_word_overlap(query, text)
        scored.append((text, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def format_structured_input(query, evidence_texts):
    """Preserved original '[SEP]' formatting."""
    return " [SEP] ".join([query, *evidence_texts])


def process_instance(inst, top_k=10, max_chars=300):
    """Preserved model-facing output fields and formatting."""
    query = inst["input_text"]
    evidence_units = inst.get("evidence_units", [])
    top_evidence = select_top_k_evidence(query, evidence_units,
                                          k=top_k, max_chars=max_chars)
    evidence_texts = [text for text, _ in top_evidence]
    return {
        "input_text": query,
        "structured_input": format_structured_input(query, evidence_texts),
        "plain_input": query + " " + " ".join(evidence_texts),
        "evidence_state_label": inst["evidence_state_label"],
        "graph_features": inst.get("graph_features", {}),
        "dataset": inst.get("dataset", "unknown"),
        "example_id": inst.get("example_id", ""),
        "num_evidence_selected": len(evidence_texts),
        "num_evidence_total": len(evidence_units),
        "target_answer": inst.get("target_answer", ""),
        "target_label": inst.get("target_label", ""),
    }


def normalized_question(inst):
    """Global grouping key, rather than split-local ordinal example IDs.

    Conservatively normalizes Unicode, capitalization, punctuation and
    whitespace. Does NOT catch semantic paraphrases or source-family sharing.
    """
    query = inst.get("input_text")
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"Missing input_text for example_id={inst.get('example_id')!r}")
    query = unicodedata.normalize("NFKC", query).casefold()
    return re.sub(r"[^\w]+", " ", query, flags=re.UNICODE).strip()


def question_hash(key):
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def stratified_split(data, test_ratio, seed):
    """Split by original question families while approximating class strata.

    Every state variant with the same normalized query stays in one split.
    Stratification is by the *full dataset/state signature of each family*,
    not by individual variant labels. A small stratum may not have test cases.

    Returns (train_rows, test_rows) as in the old function.
    """
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1 (exclusive)")

    rng = random.Random(seed)
    families = defaultdict(list)
    for inst in data:
        families[normalized_question(inst)].append(inst)

    if len(families) < 2:
        raise ValueError("Need at least two distinct question families")

    # A family may contain 1, 2, 3 or 4 states, or occasionally cross datasets.
    # Using the full signature approximates the original per-class balance.
    strata = defaultdict(list)
    for key, rows in families.items():
        signature = tuple(sorted({
            (str(x.get("dataset", "unknown")), str(x["evidence_state_label"]))
            for x in rows
        }))
        strata[signature].append(key)

    test_keys = set()
    for signature in sorted(strata):
        keys = sorted(strata[signature])
        rng.shuffle(keys)
        # For small strata, do not put the only available family into test.
        n = len(keys)
        n_test = 0 if n == 1 else min(n - 1, max(1, round(n * test_ratio)))
        test_keys.update(keys[:n_test])

    if not test_keys or len(test_keys) == len(families):
        raise ValueError("Group split did not produce nonempty train and test")

    train_split, test_split = [], []
    for key, rows in families.items():
        (test_split if key in test_keys else train_split).extend(rows)

    rng.shuffle(train_split)
    rng.shuffle(test_split)
    return train_split, test_split


def overlap_audit(train, val, test):
    """Assert question-family and exact-input separation before saving."""
    splits = {"train": train, "val": val, "test": test}
    query_sets = {
        split: {normalized_question(row) for row in rows}
        for split, rows in splits.items()
    }
    input_sets = {
        split: {row["structured_input"] for row in rows}
        for split, rows in splits.items()
    }
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    overlap_results = {}
    for left, right in pairs:
        query_intersection = query_sets[left] & query_sets[right]
        input_intersection = input_sets[left] & input_sets[right]
        overlap_results[f"{left}_vs_{right}"] = {
            "shared_normalized_questions": len(query_intersection),
            "shared_exact_structured_inputs": len(input_intersection),
            "sample_shared_question_hashes": [
                question_hash(x)[:16]
                for x in sorted(query_intersection)[:10]
            ],
        }
    report = {
        "split_strategy": "group-aware normalized-question; family-signature stratification",
        "limitations": [
            "Not a split-before-variant-generation protocol; source inputs already contain variants.",
            "Normalized exact question equality does not detect semantic paraphrases.",
            "Shared source documents/temporal families are not checked here.",
            "Feature construction artifacts and provenance truthfulness remain unaddressed.",
            "Previously inspected test examples cannot be made historically untouched.",
        ],
        "counts": {
            split: {
                "rows": len(rows),
                "question_families": len(query_sets[split]),
                "by_dataset_and_state": {
                    f"{d} / {label}": count for (d, label), count in sorted(
                        Counter((r["dataset"], r["evidence_state_label"])
                                for r in rows).items()
                    )
                },
            }
            for split, rows in splits.items()
        },
        "overlap": overlap_results,
    }
    problematic = [
        f"{pair}: {info['shared_normalized_questions']} shared questions, "
        f"{info['shared_exact_structured_inputs']} shared inputs"
        for pair, info in overlap_results.items()
        if info["shared_normalized_questions"] or info["shared_exact_structured_inputs"]
    ]
    if problematic:
        raise ValueError("Cross-split leakage detected! No outputs written. " +
                         "; ".join(problematic))
    return report


def save_split(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  Saved {len(data)} instances to {path} ({size_mb:.1f} MB)")


def print_stats(data, name):
    labels = Counter(d["evidence_state_label"] for d in data)
    datasets = Counter(d["dataset"] for d in data)
    print(f"\n  {name}:")
    print(f"    Total: {len(data)}")
    print(f"    Classes: {dict(sorted(labels.items()))}")
    print(f"    Datasets: {dict(sorted(datasets.items()))}")
    if not data:
        return
    avg_selected = sum(d["num_evidence_selected"] for d in data) / len(data)
    avg_total = sum(d["num_evidence_total"] for d in data) / len(data)
    print(f"    Avg evidence selected: {avg_selected:.1f} / {avg_total:.1f}")
    print(f"    With target_answer: {sum(1 for d in data if d.get('target_answer'))}")
    print(f"    With target_label: {sum(1 for d in data if d.get('target_label'))}")


def main():
    args = parse_args()
    if not 0 < args.test_ratio < 1:
        raise SystemExit("--test_ratio must be between 0 and 1")
    if args.top_k < 1 or args.max_evidence_chars < 1:
        raise SystemExit("--top_k and --max_evidence_chars must be positive")

    output_dir = Path(args.output_dir)
    # Safe-by-default: never overwrite the original (or any existing) outputs.
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. "
            "Choose a new revision output directory to protect existing results."
        )

    print("=" * 60)
    print("PROVE-RAG Data Preprocessing v2c — question-group split")
    print(f"Top-k evidence: {args.top_k}; Test ratio: {args.test_ratio}")
    print(f"Seed: {args.seed}")
    print("=" * 60)

    with open(args.train_json, "r", encoding="utf-8") as f:
        train_raw = json.load(f)
    with open(args.val_json, "r", encoding="utf-8") as f:
        val_raw = json.load(f)
    print(f"Loaded training-pool variants: {len(train_raw)}; validation variants: {len(val_raw)}")

    # Check the upstream validation set before any split or processing.
    train_questions = {normalized_question(x) for x in train_raw}
    val_questions = {normalized_question(x) for x in val_raw}
    if train_questions & val_questions:
        collision_hashes = [question_hash(x)[:16]
                            for x in sorted(train_questions & val_questions)[:10]]
        raise ValueError("Upstream validation questions overlap training pool; "
                         "must repair upstream source partitioning. "
                         f"Sample question hashes: {collision_hashes}")

    print("[1] Group-aware splitting of original question families...")
    train_grouped, test_grouped = stratified_split(train_raw, args.test_ratio, args.seed)
    print(f"    train variants={len(train_grouped)}, test variants={len(test_grouped)}")

    print("[2] Retaining original model input preprocessing...")
    train_final = [process_instance(r, args.top_k, args.max_evidence_chars)
                   for r in train_grouped]
    val_final = [process_instance(r, args.top_k, args.max_evidence_chars)
                 for r in val_raw]
    test_final = [process_instance(r, args.top_k, args.max_evidence_chars)
                  for r in test_grouped]
    del train_grouped, test_grouped, train_raw, val_raw

    print("[3] Auditing cross-split overlap BEFORE saving...")
    report = overlap_audit(train_final, val_final, test_final)
    print("    PASS: zero normalized-question and exact-input overlaps")

    for name, rows in (("Train", train_final), ("Val", val_final), ("Test", test_final)):
        print_stats(rows, name)

    print("[4] Saving new processed data (original outputs untouched)...")
    output_dir.mkdir(parents=True, exist_ok=True)
    save_split(train_final, output_dir / "train.json")
    save_split(val_final, output_dir / "val.json")
    save_split(test_final, output_dir / "test.json")

    with open(output_dir / "split_audit.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    # One row per question family for reproducible partitioning.
    with open(output_dir / "group_manifest.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["question_hash_sha256", "split", "variant_rows", "datasets", "states"])
        for split, rows in (("train", train_final), ("val", val_final), ("test", test_final)):
            family_stats = defaultdict(list)
            for row in rows:
                family_stats[normalized_question(row)].append(row)
            for key, instances in sorted(family_stats.items()):
                writer.writerow([
                    question_hash(key), split, len(instances),
                    ";".join(sorted({r["dataset"] for r in instances})),
                    ";".join(sorted({r["evidence_state_label"] for r in instances})),
                ])
    print(f"PASS — wrote train.json, val.json, test.json, split_audit.json, "
          f"group_manifest.csv to {output_dir}")


if __name__ == "__main__":
    main()
