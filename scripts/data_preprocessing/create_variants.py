"""
create_variants.py
==================
Takes schema-converted JSON files (sufficient/gold_full instances)
and creates evidence-state variants for training the 4-class detector.

Usage:
    # Create all variants from HotpotQA (with subset + seed)
    python create_variants.py --input data/hotpotqa_train_schema.json --variant all --output_dir data/variants/ --max_examples 5000 --seed 42

    # Create only insufficient variants
    python create_variants.py --input data/hotpotqa_train_schema.json --variant insufficient --output data/variants/hotpotqa_insufficient.json --seed 42

    # Create all variants from MuSiQue
    python create_variants.py --input data/musique_train_schema.json --variant all --output_dir data/variants/ --seed 42
"""

import argparse
import json
import os
import copy
import random
from typing import Dict, List, Optional
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create evidence-state variants from sufficient schema instances."
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to the schema JSON file (gold_full/sufficient instances)"
    )
    parser.add_argument(
        "--variant", type=str, required=True,
        choices=["insufficient", "contradicted", "superseded", "all"],
        help="Which variant to create"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for single variant (used when --variant is not 'all')"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for all variants (used when --variant is 'all')"
    )
    parser.add_argument(
        "--max_examples", type=int, default=None,
        help="Max number of sufficient instances to create variants from. "
             "If the file is too large, use this to subsample. "
             "Examples are randomly selected using --seed."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (affects both subsampling and variant creation)"
    )
    parser.add_argument(
        "--pretty", action="store_true",
        help="Pretty-print output JSON"
    )
    return parser.parse_args()


# ============================================================
# DATA LOADING WITH MEMORY SAFETY
# ============================================================

def load_instances(input_path: str, max_examples: Optional[int], seed: int) -> List[Dict]:
    """
    Load schema instances from JSON, optionally subsampling.
    
    If --max_examples is set, randomly samples that many
    instances using the provided seed for reproducibility.
    """
    file_size_mb = os.path.getsize(input_path) / (1024 * 1024)
    print(f"  File size: {file_size_mb:.1f} MB")
    
    if file_size_mb > 500 and max_examples is None:
        print(f"  WARNING: File is {file_size_mb:.0f} MB. Consider using --max_examples to subsample.")
    
    with open(input_path, "r", encoding="utf-8") as f:
        instances = json.load(f)
    
    print(f"  Loaded {len(instances)} total instances")
    
    # Filter to sufficient only
    sufficient = [
        inst for inst in instances
        if inst.get("evidence_state_label") == "sufficient"
    ]
    print(f"  {len(sufficient)} sufficient instances")
    
    # Randomly subsample if needed (seeded for reproducibility)
    if max_examples and max_examples < len(sufficient):
        rng = random.Random(seed)
        sufficient = rng.sample(sufficient, max_examples)
        print(f"  Subsampled to {len(sufficient)} instances (seed={seed})")
    
    return sufficient


# ============================================================
# VARIANT 1: INSUFFICIENT
# ============================================================

def create_insufficient(instance: Dict, rng: random.Random) -> Optional[Dict]:
    """
    Create an insufficient variant by removing gold evidence.
    
    For multi-hop QA, this simulates a missing reasoning hop.
    The question remains the same but can no longer be fully
    answered from the remaining evidence.
    
    Example:
        Q: "What year did the director of Titanic win an Oscar?"
        Gold: [A: "Titanic directed by James Cameron",
               B: "James Cameron won Oscar in 1998"]
        Insufficient: Remove B -> can't answer from A alone
    """
    variant = copy.deepcopy(instance)
    
    gold_indices = [
        i for i, eu in enumerate(variant["evidence_units"])
        if eu["is_gold_evidence"]
    ]
    
    if len(gold_indices) == 0:
        return None
    
    # Remove exactly 1 gold unit
    remove_indices = set(rng.sample(gold_indices, 1))
    
    variant["evidence_units"] = [
        eu for i, eu in enumerate(variant["evidence_units"])
        if i not in remove_indices
    ]
    
    # Update labels
    variant["evidence_state_label"] = "insufficient"
    variant["agent_action_label"] = "retrieve_more"
    
    # Update example ID
    old_id = instance["example_id"]
    variant["example_id"] = old_id.replace("gold_full", "insufficient")
    if variant["example_id"] == old_id:
        variant["example_id"] = old_id.replace("sufficient", "insufficient")
    if variant["example_id"] == old_id:
        variant["example_id"] = old_id + "_insufficient"
    
    # Recompute stats
    variant["evidence_set"]["condition"] = "missing_evidence"
    variant["evidence_set"]["num_evidence_units"] = len(variant["evidence_units"])
    variant["evidence_set"]["num_gold_evidence_units"] = sum(
        1 for eu in variant["evidence_units"] if eu["is_gold_evidence"]
    )
    variant["evidence_set"]["num_sources"] = len(
        set(eu["doc_title"] for eu in variant["evidence_units"])
    )
    
    return variant


# ============================================================
# VARIANT 2: CONTRADICTED
# ============================================================

def create_contradicted(instance: Dict, rng: random.Random) -> Optional[Dict]:
    """
    Create a contradicted variant by adding conflicting evidence.
    
    Keeps all original evidence intact and ADDS a new unit that
    contradicts one of the gold units. The evidence set now
    contains conflicting information from different "sources."
    
    Example:
        Q: "Where was Einstein born?"
        Gold: ["Einstein was born in Ulm, Germany"]
        Contradicted: Gold + ["Recent records show Einstein
                               was born in Munich"] 
        -> Conflict in evidence set
    """
    variant = copy.deepcopy(instance)
    
    gold_units = [
        (i, eu) for i, eu in enumerate(variant["evidence_units"])
        if eu["is_gold_evidence"] and eu.get("text")
    ]
    
    if len(gold_units) == 0:
        return None
    
    target_idx, target_unit = rng.choice(gold_units)
    original_text = target_unit["text"]
    
    contradicted_text = negate_text(original_text, instance.get("target_answer"), rng)
    
    # Create NEW contradicting unit (don't replace original)
    contradicting_unit = copy.deepcopy(target_unit)
    contradicting_unit["text"] = contradicted_text
    contradicting_unit["evidence_id"] = target_unit["evidence_id"] + "_contradicting"
    contradicting_unit["is_gold_evidence"] = False
    contradicting_unit["support_role"] = "refutes"
    contradicting_unit["native_label"] = "injected_contradiction"
    contradicting_unit["label_strength"] = "synthetic"
    contradicting_unit["supervision_weight"] = 0.0
    contradicting_unit["provenance"]["source_type"] = "synthetic_contradiction"
    
    variant["evidence_units"].append(contradicting_unit)
    
    # Update labels
    variant["evidence_state_label"] = "contradicted"
    variant["agent_action_label"] = "flag_conflict"
    
    old_id = instance["example_id"]
    variant["example_id"] = old_id.replace("gold_full", "contradicted")
    if variant["example_id"] == old_id:
        variant["example_id"] = old_id.replace("sufficient", "contradicted")
    if variant["example_id"] == old_id:
        variant["example_id"] = old_id + "_contradicted"
    
    variant["evidence_set"]["condition"] = "contains_contradiction"
    variant["evidence_set"]["num_evidence_units"] = len(variant["evidence_units"])
    variant["evidence_set"]["num_sources"] = len(
        set(eu["doc_title"] for eu in variant["evidence_units"])
    )
    
    return variant


def negate_text(text: str, answer: Optional[str], rng: random.Random) -> str:
    """
    Generate a contradicting version of evidence text using templates.
    
    For the final paper, upgrade to LLM-generated contradictions.
    For report, templates are sufficient.
    """
    short_text = text[:120].rstrip()
    if len(text) > 120:
        short_text += "..."
    
    general_templates = [
        f"Contrary to some sources, {short_text.lower()} This claim has been disputed by multiple independent researchers and is considered unreliable.",
        f"However, recent findings contradict this. The assertion that {short_text.lower()} has been shown to be inaccurate based on newly available evidence.",
        f"This information is no longer considered accurate. Updated records show that the claim regarding {short_text.lower()} was based on incomplete data.",
        f"Independent verification has failed to confirm that {short_text.lower()} Multiple fact-checking organizations have flagged this as unsubstantiated.",
    ]
    
    if answer and len(answer) > 1:
        targeted_templates = [
            f"Despite earlier reports, {answer} has been shown to be incorrect in this context. The actual facts differ significantly from what was previously claimed.",
            f"Records have been updated to reflect that the information involving {answer} was erroneous. The corrected version contradicts the original claim.",
            f"It has been established that the reference to {answer} in this context is factually wrong, based on more recent and reliable sources.",
        ]
        all_templates = general_templates + targeted_templates
    else:
        all_templates = general_templates
    
    return rng.choice(all_templates)


# ============================================================
# VARIANT 3: SUPERSEDED
# ============================================================

def create_superseded(instance: Dict, rng: random.Random) -> Optional[Dict]:
    """
    Create a superseded variant with outdated + newer evidence.
    
    Marks original gold evidence as outdated (old timestamp)
    and adds a "newer" version with a recent timestamp that
    contradicts the original. Both versions remain in the set.
    
    Example:
        Q: "Who is the CEO of Twitter?"
        Gold (old): ["Jack Dorsey is CEO" timestamp: 2020]
        Superseded: Gold + ["Elon Musk became CEO" timestamp: 2023]
        -> Need to verify which is current
    """
    variant = copy.deepcopy(instance)
    
    gold_units = [
        (i, eu) for i, eu in enumerate(variant["evidence_units"])
        if eu["is_gold_evidence"] and eu.get("text")
    ]
    
    if len(gold_units) == 0:
        return None
    
    target_idx, target_unit = rng.choice(gold_units)
    short_text = target_unit["text"][:120].rstrip()
    
    # Mark original as outdated
    variant["evidence_units"][target_idx]["provenance"]["timestamp"] = "2018-01-01"
    variant["evidence_units"][target_idx]["provenance"]["version"] = "v1_outdated"
    
    supersession_templates = [
        f"[Updated 2025] Previous information stating that {short_text.lower()}... has been revised. Current records reflect different facts.",
        f"[Correction - 2025] The earlier claim that {short_text.lower()}... has been superseded by more recent findings.",
        f"[2025 Update] This information has changed. The original statement regarding {short_text.lower()}... no longer reflects the current state of affairs.",
    ]
    
    # Create newer version
    newer_unit = copy.deepcopy(target_unit)
    newer_unit["text"] = rng.choice(supersession_templates)
    newer_unit["evidence_id"] = target_unit["evidence_id"] + "_v2_current"
    newer_unit["is_gold_evidence"] = False
    newer_unit["support_role"] = "supersedes"
    newer_unit["native_label"] = "newer_version"
    newer_unit["label_strength"] = "synthetic"
    newer_unit["supervision_weight"] = 0.0
    newer_unit["provenance"]["timestamp"] = "2025-01-01"
    newer_unit["provenance"]["version"] = "v2_current"
    newer_unit["provenance"]["source_type"] = "synthetic_supersession"
    
    variant["evidence_units"].append(newer_unit)
    
    # Update labels
    variant["evidence_state_label"] = "superseded"
    variant["agent_action_label"] = "verify"
    
    old_id = instance["example_id"]
    variant["example_id"] = old_id.replace("gold_full", "superseded")
    if variant["example_id"] == old_id:
        variant["example_id"] = old_id.replace("sufficient", "superseded")
    if variant["example_id"] == old_id:
        variant["example_id"] = old_id + "_superseded"
    
    variant["evidence_set"]["condition"] = "contains_outdated"
    variant["evidence_set"]["num_evidence_units"] = len(variant["evidence_units"])
    variant["evidence_set"]["num_sources"] = len(
        set(eu["doc_title"] for eu in variant["evidence_units"])
    )
    
    return variant


# ============================================================
# MAIN
# ============================================================

def run(args):
    random.seed(args.seed)
    rng = random.Random(args.seed)
    
    print(f"{'='*60}")
    print(f"Creating evidence-state variants")
    print(f"Seed: {args.seed}")
    print(f"{'='*60}")
    
    # Load and optionally subsample
    print(f"\nLoading {args.input}...")
    sufficient = load_instances(args.input, args.max_examples, args.seed)
    
    if len(sufficient) == 0:
        print("ERROR: No sufficient instances found.")
        return
    
    # Map variant names to functions
    creators = {
        "insufficient": create_insufficient,
        "contradicted": create_contradicted,
        "superseded": create_superseded,
    }
    
    variant_types = ["insufficient", "contradicted", "superseded"] if args.variant == "all" else [args.variant]
    
    for variant_type in variant_types:
        print(f"\n--- Creating {variant_type} variants ---")
        creator = creators[variant_type]
        
        # Separate RNG per variant type for independence
        variant_rng = random.Random(args.seed + hash(variant_type) % 10000)
        
        results = []
        skipped = 0
        
        for inst in sufficient:
            result = creator(inst, variant_rng)
            if result:
                results.append(result)
            else:
                skipped += 1
        
        print(f"  Created: {len(results)}")
        print(f"  Skipped: {skipped}")
        
        # Determine output path
        if args.variant == "all":
            out_dir = args.output_dir or "data/variants"
            os.makedirs(out_dir, exist_ok=True)
            input_basename = os.path.splitext(os.path.basename(args.input))[0]
            output_path = os.path.join(out_dir, f"{input_basename}_{variant_type}.json")
        else:
            output_path = args.output
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        # Save
        with open(output_path, "w", encoding="utf-8") as f:
            if args.pretty:
                json.dump(results, f, indent=2, ensure_ascii=False)
            else:
                json.dump(results, f, ensure_ascii=False)
        
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  Saved to {output_path} ({file_size:.1f} MB)")
        
        # Show a sample
        if results:
            s = results[0]
            print(f"  Sample: ID={s['example_id']}")
            print(f"          State={s['evidence_state_label']}, Action={s['agent_action_label']}")
            print(f"          Evidence={s['evidence_set']['num_evidence_units']}, Gold={s['evidence_set']['num_gold_evidence_units']}")
    
    # Summary
    print(f"\n{'='*60}")
    print("DONE! Next steps:")
    print("  1. Run the same command for validation split")
    print("  2. Merge sufficient + variants into unified train/val files")
    print("  3. Build evidence graphs (evidence_graph.py)")
    print(f"{'='*60}")


if __name__ == "__main__":
    args = parse_args()
    run(args)
