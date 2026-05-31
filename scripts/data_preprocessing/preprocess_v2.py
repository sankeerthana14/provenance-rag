"""
preprocess_v2.py
=================
Creates publication-ready data splits with structured evidence formatting.

Changes from v1:
1. Splits train into train (80%) + test (20%), keeps val as-is
2. Selects top-k evidence units by query relevance (word overlap)
3. Formats evidence with [SEP] markers for structured input
4. Creates lightweight JSON files that load in seconds

Changes in v2b:
5. Preserves target_answer and target_label for end-to-end evaluation

Usage:
    python preprocess_v2.py \
        --train_json results/unified_train.json \
        --val_json results/unified_val.json \
        --output_dir data/processed/ \
        --top_k 10 \
        --seed 42
"""

import json
import os
import argparse
import random
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", type=str, required=True)
    parser.add_argument("--val_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=10,
                        help="Max evidence units to include per instance")
    parser.add_argument("--test_ratio", type=float, default=0.2,
                        help="Fraction of train to hold out as test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_evidence_chars", type=int, default=300,
                        help="Max characters per evidence unit")
    return parser.parse_args()


def compute_word_overlap(query, text):
    """Score evidence relevance by word overlap with query."""
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
    """
    Select top-k evidence units by relevance to query.
    Returns list of (text, score) tuples.
    """
    scored = []
    for eu in evidence_units:
        text = eu.get("text", "")
        if not text or not text.strip():
            continue
        text = text.strip()[:max_chars]
        score = compute_word_overlap(query, text)
        scored.append((text, score))
    
    # Sort by relevance, take top-k
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def format_structured_input(query, evidence_texts):
    """
    Create structured input: query [SEP] evidence_1 [SEP] evidence_2 ...
    The tokenizer will convert [SEP] to actual separator tokens.
    """
    parts = [query]
    for text in evidence_texts:
        parts.append(text)
    return " [SEP] ".join(parts)


def process_instance(inst, top_k=10, max_chars=300):
    """Convert one instance to lightweight format with structured input."""
    query = inst["input_text"]
    evidence_units = inst.get("evidence_units", [])
    
    # Select top-k most relevant evidence
    top_evidence = select_top_k_evidence(query, evidence_units, k=top_k,
                                          max_chars=max_chars)
    evidence_texts = [t for t, s in top_evidence]
    
    # Create structured input
    structured_input = format_structured_input(query, evidence_texts)
    
    # Also keep plain concatenated version for comparison
    plain_input = query + " " + " ".join(evidence_texts)
    
    return {
        "input_text": query,
        "structured_input": structured_input,
        "plain_input": plain_input,
        "evidence_state_label": inst["evidence_state_label"],
        "graph_features": inst.get("graph_features", {}),
        "dataset": inst.get("dataset", "unknown"),
        "example_id": inst.get("example_id", ""),
        "num_evidence_selected": len(evidence_texts),
        "num_evidence_total": len(evidence_units),
        "target_answer": inst.get("target_answer", ""),
        "target_label": inst.get("target_label", ""),
    }


def stratified_split(data, test_ratio, seed):
    """
    Split data into train and test, stratified by both
    evidence_state_label AND dataset.
    """
    rng = random.Random(seed)
    
    # Group by (label, dataset)
    groups = {}
    for inst in data:
        key = (inst["evidence_state_label"], inst["dataset"])
        if key not in groups:
            groups[key] = []
        groups[key].append(inst)
    
    train_split = []
    test_split = []
    
    for key, instances in groups.items():
        rng.shuffle(instances)
        n_test = max(1, int(len(instances) * test_ratio))
        test_split.extend(instances[:n_test])
        train_split.extend(instances[n_test:])
    
    rng.shuffle(train_split)
    rng.shuffle(test_split)
    
    return train_split, test_split


def save_split(data, path):
    """Save processed data as JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  Saved {len(data)} instances to {path} ({size_mb:.1f} MB)")


def print_stats(data, name):
    """Print class and dataset distribution."""
    labels = Counter(d["evidence_state_label"] for d in data)
    datasets = Counter(d["dataset"] for d in data)
    
    print(f"\n  {name}:")
    print(f"    Total: {len(data)}")
    print(f"    Classes: {dict(sorted(labels.items()))}")
    print(f"    Datasets: {dict(sorted(datasets.items()))}")
    
    # Evidence selection stats
    avg_selected = sum(d["num_evidence_selected"] for d in data) / len(data)
    avg_total = sum(d["num_evidence_total"] for d in data) / len(data)
    print(f"    Avg evidence selected: {avg_selected:.1f} / {avg_total:.1f}")
    
    # Gold answer stats
    has_answer = sum(1 for d in data if d.get("target_answer"))
    has_label = sum(1 for d in data if d.get("target_label"))
    print(f"    With target_answer: {has_answer}")
    print(f"    With target_label: {has_label}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("PROVE-RAG Data Preprocessing v2b")
    print(f"Top-k evidence: {args.top_k}")
    print(f"Test ratio: {args.test_ratio}")
    print(f"Seed: {args.seed}")
    print("=" * 60)
    
    # Load train data
    print(f"\n[1] Loading train data from {args.train_json}...")
    with open(args.train_json, "r", encoding="utf-8") as f:
        train_raw = json.load(f)
    print(f"  Loaded {len(train_raw)} instances")
    
    # Process train instances
    print(f"\n[2] Processing instances (top-{args.top_k} evidence selection)...")
    train_processed = []
    for i, inst in enumerate(train_raw):
        train_processed.append(process_instance(inst, args.top_k, args.max_evidence_chars))
        if (i + 1) % 50000 == 0:
            print(f"    Processed {i + 1}/{len(train_raw)}")
    del train_raw
    print(f"  Processed {len(train_processed)} train instances")
    
    # Split train into train + test
    print(f"\n[3] Splitting train into train ({1-args.test_ratio:.0%}) + test ({args.test_ratio:.0%})...")
    train_final, test_final = stratified_split(train_processed, args.test_ratio, args.seed)
    del train_processed
    
    # Load and process val data
    print(f"\n[4] Loading and processing val data...")
    with open(args.val_json, "r", encoding="utf-8") as f:
        val_raw = json.load(f)
    
    val_final = [process_instance(inst, args.top_k, args.max_evidence_chars)
                 for inst in val_raw]
    del val_raw
    
    # Print statistics
    print(f"\n[5] Split statistics:")
    print_stats(train_final, "Train")
    print_stats(val_final, "Val")
    print_stats(test_final, "Test")
    
    # Save
    print(f"\n[6] Saving...")
    save_split(train_final, os.path.join(args.output_dir, "train.json"))
    save_split(val_final, os.path.join(args.output_dir, "val.json"))
    save_split(test_final, os.path.join(args.output_dir, "test.json"))
    
    print(f"\n{'='*60}")
    print("DONE!")
    print(f"{'='*60}")
    print(f"\nOutputs in {args.output_dir}/:")
    print(f"  train.json - training data ({len(train_final)} instances)")
    print(f"  val.json   - validation data ({len(val_final)} instances)")
    print(f"  test.json  - held-out test data ({len(test_final)} instances)")
    print(f"\nEach instance contains:")
    print(f"  structured_input: 'query [SEP] ev1 [SEP] ev2 ...' (top-{args.top_k})")
    print(f"  plain_input: 'query ev1 ev2 ...' (for comparison)")
    print(f"  graph_features: 8 provenance features")
    print(f"  evidence_state_label: sufficient/insufficient/contradicted/superseded")
    print(f"  target_answer: gold answer (HotpotQA/MuSiQue)")
    print(f"  target_label: gold label (FEVER)")


if __name__ == "__main__":
    main()