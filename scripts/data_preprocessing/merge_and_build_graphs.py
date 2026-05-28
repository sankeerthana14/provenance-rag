"""
merge_and_build_graphs.py
==========================
1. Merges sufficient instances + variant files into unified train/val datasets
2. Builds evidence graphs per instance using NetworkX
3. Extracts 6 provenance-derived graph features (NO label leakage)

Usage:
    python merge_and_build_graphs.py \
        --data_dir C:\path\to\data\modified \
        --variants_dir C:\path\to\data\modified\variants \
        --output_dir C:\path\to\data\final \
        --seed 42

Features extracted (all label-free):
    1. source_diversity           - number of unique source documents
    2. text_resolution_rate       - fraction of evidence with resolved text
    3. avg_evidence_length        - mean token count across evidence units
    4. min_evidence_length        - min token count (captures very short/empty evidence)
    5. duplicate_rate             - fraction of near-duplicate evidence pairs (Jaccard)
    6. document_overlap_rate      - fraction of evidence units sharing a source with another unit
    7. entity_overlap             - fraction of query entities found in evidence
    8. evidence_count             - total number of evidence units

Dropped features (would cause label leakage or don't vary in benchmark data):
    - support_coverage, hop_completeness, avg_supervision_weight (use annotation labels)
    - has_refutation (uses support_role annotation)
    - retrieval_rank_distribution (no retriever in our pipeline)
    - source_domain_count (all datasets use Wikipedia)
    - timestamp_availability, evidence_recency_score (only exist for synthetic superseded)
"""

import argparse
import json
import os
import random
import csv
import networkx as nx
import numpy as np
from collections import Counter
from typing import Dict, List, Optional, Set


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge datasets + build evidence graphs with clean features"
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing schema JSON files")
    parser.add_argument("--variants_dir", type=str, required=True,
                        help="Directory containing variant JSON files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for merged files + features")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balance", action="store_true", default=True,
                        help="Balance classes by subsampling to smallest class")
    parser.add_argument("--skip_ner", action="store_true",
                        help="Skip entity overlap computation (faster, no spaCy needed)")
    parser.add_argument("--pretty", action="store_true",
                        help="Pretty-print output JSON")
    return parser.parse_args()


# ============================================================
# STEP 1: MERGE
# ============================================================

def load_json(path):
    """Load a JSON file with UTF-8 encoding."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_file(directory, pattern):
    """Find a file matching a pattern in a directory."""
    for f in os.listdir(directory):
        if pattern in f and f.endswith(".json"):
            return os.path.join(directory, f)
    return None


def merge_split(data_dir, variants_dir, split, seed, balance=True):
    """
    Merge sufficient instances + variants for one split (train or val).
    
    CRITICAL: Only take sufficient instances from schema files.
    - MuSiQue schema has natural insufficient (unanswerable) — excluded
    - FEVER schema has natural contradicted (REFUTES) and insufficient (NEI) — excluded
    - Only synthetic variants from create_variants.py are used for other classes
    """
    rng = random.Random(seed)
    
    # Define file patterns for each dataset and split
    if split == "train":
        schema_patterns = {
            "hotpotqa": "hotpotqa_train_15k",
            "musique": "musique_train_30k",
            "fever": "fever_train_30k",
        }
    else:  # val
        schema_patterns = {
            "hotpotqa": "hotpotqa_val_2.5k",
            "musique": "musique_val_5k",
            "fever": "fever_val_7k",
        }
    
    all_instances = {
        "sufficient": [],
        "insufficient": [],
        "contradicted": [],
        "superseded": [],
    }
    
    for dataset_name, pattern in schema_patterns.items():
        print(f"\n  --- {dataset_name} ---")
        
        # Load sufficient instances from schema file
        schema_path = find_file(data_dir, pattern)
        if schema_path:
            print(f"  Schema: {os.path.basename(schema_path)}")
            schema_data = load_json(schema_path)
            sufficient = [
                inst for inst in schema_data
                if inst.get("evidence_state_label") == "sufficient"
            ]
            all_instances["sufficient"].extend(sufficient)
            print(f"    Sufficient: {len(sufficient)}")
        else:
            print(f"  WARNING: Schema file not found for {dataset_name} ({pattern})")
        
        # Load variants from variant files
        for variant_type in ["insufficient", "contradicted", "superseded"]:
            variant_path = find_file(variants_dir, f"{pattern}_{variant_type}")
            if variant_path:
                variant_data = load_json(variant_path)
                all_instances[variant_type].extend(variant_data)
                print(f"    {variant_type}: {len(variant_data)}")
            else:
                print(f"    WARNING: {variant_type} variant not found for {pattern}")
    
    # Print pre-balance distribution
    print(f"\n  Pre-balance distribution:")
    for label, instances in all_instances.items():
        print(f"    {label}: {len(instances)}")
    
    # Balance classes by subsampling to smallest class
    if balance:
        min_count = min(len(v) for v in all_instances.values())
        print(f"\n  Balancing to {min_count} per class...")
        for label in all_instances:
            if len(all_instances[label]) > min_count:
                all_instances[label] = rng.sample(all_instances[label], min_count)
    
    # Merge and shuffle
    merged = []
    for label, instances in all_instances.items():
        merged.extend(instances)
    rng.shuffle(merged)
    
    # Print final distribution
    print(f"\n  Final distribution:")
    dist = Counter(inst["evidence_state_label"] for inst in merged)
    for label, count in sorted(dist.items()):
        print(f"    {label}: {count}")
    print(f"    Total: {len(merged)}")
    
    # Per-dataset breakdown
    dataset_dist = Counter(inst["dataset"] for inst in merged)
    print(f"\n  Per-dataset breakdown:")
    for ds, count in sorted(dataset_dist.items()):
        print(f"    {ds}: {count}")
    
    return merged


# ============================================================
# STEP 2: EVIDENCE GRAPHS
# ============================================================

def build_evidence_graph(instance: Dict) -> nx.DiGraph:
    """
    Build a directed evidence graph for one instance.
    
    This graph uses ONLY structural/textual information,
    not annotation labels. Edges are based on:
    - Source attribution (which document does evidence come from)
    - Text similarity (do evidence units overlap significantly)
    - Temporal relationships (do timestamps conflict)
    """
    G = nx.DiGraph()
    
    # Add input node
    G.add_node("input",
               type="input",
               text=instance["input_text"][:200])
    
    # Track source documents
    sources = set()
    
    for eu in instance["evidence_units"]:
        ev_id = eu["evidence_id"]
        doc_title = eu["doc_title"]
        
        # Add evidence node (no annotation fields stored)
        G.add_node(ev_id,
                   type="evidence",
                   has_text=eu.get("text") is not None and eu.get("text", "") != "",
                   doc_title=doc_title)
        
        # Edge: evidence → input (all evidence relates to the input)
        G.add_edge(ev_id, "input", relation="relates_to")
        
        # Add source document node + attribution edge
        source_id = eu.get("source_doc_id", f"source::{doc_title}")
        if source_id not in sources:
            G.add_node(source_id, type="source", title=doc_title)
            sources.add(source_id)
        G.add_edge(ev_id, source_id, relation="attribution")
    
    # Add text-similarity edges between evidence units that share
    # significant token overlap (computed without labels)
    evidence_with_text = [
        eu for eu in instance["evidence_units"]
        if eu.get("text") and eu["text"].strip()
    ]
    
    for i in range(len(evidence_with_text)):
        for j in range(i + 1, len(evidence_with_text)):
            sim = jaccard_similarity(
                evidence_with_text[i]["text"],
                evidence_with_text[j]["text"]
            )
            if sim > 0.3:  # Threshold for "similar enough to note"
                G.add_edge(
                    evidence_with_text[i]["evidence_id"],
                    evidence_with_text[j]["evidence_id"],
                    relation="text_overlap",
                    similarity=sim
                )
    
    return G


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """
    Compute Jaccard similarity between two texts based on word tokens.
    Returns a value between 0 and 1.
    No labels needed — purely text-based.
    """
    if not text_a or not text_b:
        return 0.0
    
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    
    if not tokens_a or not tokens_b:
        return 0.0
    
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    
    return len(intersection) / len(union) if union else 0.0


# ============================================================
# STEP 3: FEATURE EXTRACTION (NO LABEL LEAKAGE)
# ============================================================

def extract_graph_features(
    instance: Dict,
    graph: nx.DiGraph,
    nlp=None  # spaCy model, optional
) -> Dict:
    """
    Extract 8 provenance-derived features from the evidence graph.
    
    NONE of these use annotation fields (is_gold_evidence, 
    support_role, supervision_weight, label_strength).
    All are computable from raw text + source metadata.
    """
    evidence_units = instance["evidence_units"]
    total_units = len(evidence_units)
    
    # --- Feature 1: source_diversity ---
    # Number of unique source documents.
    # More diverse sources = better corroboration potential.
    # Computable from: doc_title (metadata, not label)
    unique_sources = set(
        eu["doc_title"] for eu in evidence_units
        if eu["doc_title"] and eu["doc_title"] != "none"
    )
    source_diversity = len(unique_sources)
    
    # --- Feature 2: text_resolution_rate ---
    # Fraction of evidence units that have actual text content.
    # Some FEVER evidence is unresolved (text=None).
    # Computable from: text field presence (not a label)
    text_available = sum(
        1 for eu in evidence_units
        if eu.get("text") and eu["text"].strip()
    )
    text_resolution_rate = text_available / max(total_units, 1)
    
    # --- Feature 3: avg_evidence_length ---
    # Average number of tokens across evidence units.
    # Shorter evidence may be less informative.
    # Computable from: text content (not a label)
    lengths = []
    for eu in evidence_units:
        if eu.get("text") and eu["text"].strip():
            lengths.append(len(eu["text"].split()))
    avg_evidence_length = np.mean(lengths) if lengths else 0.0
    
    # --- Feature 4: min_evidence_length ---
    # Minimum evidence length. Very short evidence units
    # (or empty ones) may indicate resolution failures.
    min_evidence_length = min(lengths) if lengths else 0.0
    
    # --- Feature 5: duplicate_rate ---
    # Fraction of evidence pairs with high text overlap (Jaccard > 0.5).
    # High duplication = redundant retrieval, potentially suspicious.
    # Computable from: text content (not a label)
    texts = [
        eu["text"] for eu in evidence_units
        if eu.get("text") and eu["text"].strip()
    ]
    num_pairs = 0
    num_duplicates = 0
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            num_pairs += 1
            if jaccard_similarity(texts[i], texts[j]) > 0.5:
                num_duplicates += 1
    duplicate_rate = num_duplicates / max(num_pairs, 1)
    
    # --- Feature 6: document_overlap_rate ---
    # Fraction of evidence units that share a source document
    # with at least one other unit. High overlap = evidence
    # clustered in few sources rather than spread across many.
    # Computable from: doc_title (metadata, not a label)
    doc_counts = Counter(
        eu["doc_title"] for eu in evidence_units
        if eu["doc_title"] and eu["doc_title"] != "none"
    )
    units_sharing_source = sum(
        1 for eu in evidence_units
        if eu["doc_title"] and doc_counts.get(eu["doc_title"], 0) > 1
    )
    document_overlap_rate = units_sharing_source / max(total_units, 1)
    
    # --- Feature 7: entity_overlap ---
    # Fraction of named entities in the query that also appear
    # in the evidence. Low overlap = evidence may not cover
    # the query's key entities.
    # Computable from: NER on text (not a label)
    if nlp is not None:
        entity_overlap = compute_entity_overlap(
            instance["input_text"], evidence_units, nlp
        )
    else:
        # Fallback: simple word overlap between query and evidence
        entity_overlap = compute_word_overlap(
            instance["input_text"], evidence_units
        )
    
    # --- Feature 8: evidence_count ---
    # Total number of evidence units. More evidence isn't always
    # better (could include noise), but zero/low is informative.
    # Computable from: count (not a label)
    evidence_count = total_units
    
    return {
        "source_diversity": source_diversity,
        "text_resolution_rate": round(text_resolution_rate, 4),
        "avg_evidence_length": round(avg_evidence_length, 2),
        "min_evidence_length": round(min_evidence_length, 2),
        "duplicate_rate": round(duplicate_rate, 4),
        "document_overlap_rate": round(document_overlap_rate, 4),
        "entity_overlap": round(entity_overlap, 4),
        "evidence_count": evidence_count,
    }


def compute_entity_overlap(
    query: str,
    evidence_units: List[Dict],
    nlp
) -> float:
    """
    Compute entity overlap between query and evidence using spaCy NER.
    Returns fraction of query entities found in evidence.
    """
    # Extract entities from query
    query_doc = nlp(query)
    query_entities = set(ent.text.lower() for ent in query_doc.ents)
    
    if not query_entities:
        return 0.0
    
    # Extract entities from all evidence
    evidence_text = " ".join(
        eu["text"] for eu in evidence_units
        if eu.get("text") and eu["text"].strip()
    )
    if not evidence_text.strip():
        return 0.0
    
    evidence_doc = nlp(evidence_text[:10000])  # Limit length for speed
    evidence_entities = set(ent.text.lower() for ent in evidence_doc.ents)
    
    # Compute overlap
    overlap = query_entities & evidence_entities
    return len(overlap) / len(query_entities)


def compute_word_overlap(
    query: str,
    evidence_units: List[Dict]
) -> float:
    """
    Fallback: compute word overlap between query and evidence.
    Used when spaCy is not available (--skip_ner flag).
    Filters out common stopwords for a cleaner signal.
    """
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "and",
        "but", "or", "not", "no", "if", "then", "than", "that",
        "this", "which", "who", "whom", "what", "where", "when",
        "how", "all", "each", "every", "both", "few", "more",
        "most", "other", "some", "such", "only", "own", "same",
        "so", "very", "it", "its", "they", "them", "their", "he",
        "she", "him", "her", "his", "we", "us", "our", "you", "your",
    }
    
    # Get content words from query
    query_words = set(query.lower().split()) - stopwords
    
    if not query_words:
        return 0.0
    
    # Get content words from all evidence
    evidence_text = " ".join(
        eu["text"] for eu in evidence_units
        if eu.get("text") and eu["text"].strip()
    )
    evidence_words = set(evidence_text.lower().split()) - stopwords
    
    if not evidence_words:
        return 0.0
    
    overlap = query_words & evidence_words
    return len(overlap) / len(query_words)


# ============================================================
# MAIN
# ============================================================

def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("MERGE + EVIDENCE GRAPH PIPELINE")
    print(f"Seed: {args.seed}")
    print(f"NER: {'disabled (--skip_ner)' if args.skip_ner else 'enabled (spaCy)'}")
    print("=" * 60)
    
    # Load spaCy if NER is enabled
    nlp = None
    if not args.skip_ner:
        try:
            import spacy
            print("\nLoading spaCy model (en_core_web_sm)...")
            nlp = spacy.load("en_core_web_sm")
            print("  Loaded successfully")
        except (ImportError, OSError) as e:
            print(f"  WARNING: Could not load spaCy: {e}")
            print("  Falling back to word overlap for entity_overlap feature.")
            print("  To enable NER: pip install spacy && python -m spacy download en_core_web_sm")
            nlp = None
    
    for split in ["train", "val"]:
        print(f"\n{'='*60}")
        print(f"Processing {split.upper()} split")
        print(f"{'='*60}")
        
        # Step 1: Merge
        print(f"\n[Step 1] Merging {split} data...")
        merged = merge_split(
            args.data_dir, args.variants_dir, split,
            args.seed, balance=args.balance
        )
        
        if not merged:
            print(f"  ERROR: No data merged for {split}. Check file paths.")
            continue
        
        # Step 2: Build graphs + extract features
        print(f"\n[Step 2] Building evidence graphs + extracting features...")
        feature_names = None
        all_features = []
        
        for i, instance in enumerate(merged):
            graph = build_evidence_graph(instance)
            features = extract_graph_features(instance, graph, nlp)
            
            # Attach features to the instance
            instance["graph_features"] = features
            all_features.append(features)
            
            if feature_names is None:
                feature_names = list(features.keys())
            
            if (i + 1) % 5000 == 0:
                print(f"  Processed {i + 1}/{len(merged)} instances")
        
        print(f"  Done: {len(merged)} instances with graph features")
        
        # Step 3: Print feature statistics
        print(f"\n[Step 3] Overall feature statistics:")
        for feat_name in feature_names:
            values = [f[feat_name] for f in all_features]
            print(f"  {feat_name:30s}: "
                  f"mean={np.mean(values):8.4f}, "
                  f"std={np.std(values):8.4f}, "
                  f"min={np.min(values):8.4f}, "
                  f"max={np.max(values):8.4f}")
        
        # Step 4: Print features per evidence state
        print(f"\n[Step 4] Features by evidence state:")
        for state in ["sufficient", "insufficient", "contradicted", "superseded"]:
            state_features = [
                inst["graph_features"] for inst in merged
                if inst["evidence_state_label"] == state
            ]
            if state_features:
                print(f"\n  {state} (n={len(state_features)}):")
                for feat_name in feature_names:
                    values = [f[feat_name] for f in state_features]
                    print(f"    {feat_name:30s}: mean={np.mean(values):.4f}")
        
        # Step 5: Save unified JSON
        output_path = os.path.join(args.output_dir, f"unified_{split}.json")
        print(f"\n[Step 5] Saving to {output_path}...")
        with open(output_path, "w", encoding="utf-8") as f:
            if args.pretty:
                json.dump(merged, f, indent=2, ensure_ascii=False)
            else:
                json.dump(merged, f, ensure_ascii=False)
        
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  Saved ({file_size:.1f} MB)")
        
        # Step 6: Save features CSV
        features_csv_path = os.path.join(args.output_dir, f"graph_features_{split}.csv")
        with open(features_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["example_id", "dataset", "evidence_state_label"] + feature_names
            )
            writer.writeheader()
            for inst in merged:
                row = {
                    "example_id": inst["example_id"],
                    "dataset": inst["dataset"],
                    "evidence_state_label": inst["evidence_state_label"],
                    **inst["graph_features"],
                }
                writer.writerow(row)
        print(f"  Features CSV saved to {features_csv_path}")
    
    print(f"\n{'='*60}")
    print("ALL DONE!")
    print(f"{'='*60}")
    print(f"\nOutputs in {args.output_dir}:")
    print(f"  unified_train.json  — balanced training data with graph features")
    print(f"  unified_val.json    — balanced validation data with graph features")
    print(f"  graph_features_train.csv — features only (for analysis)")
    print(f"  graph_features_val.csv   — features only (for analysis)")
    print(f"\nFeatures extracted (all label-free):")
    if feature_names:
        for fn in feature_names:
            print(f"  - {fn}")
    print(f"\nNext step: Train the evidence-state detector (DeBERTa)!")


if __name__ == "__main__":
    args = parse_args()
    run(args)