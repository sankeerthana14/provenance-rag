"""
convert_to_schema.py
====================
Converts raw datasets (HotpotQA, FEVER, MuSiQue) into the
unified provenance-aware evidence schema for PROVE-RAG.

Usage:
    python convert_to_schema.py --dataset hotpotqa --output data/hotpotqa_schema.json
    python convert_to_schema.py --dataset hotpotqa --output data/hotpotqa_schema.json --max_examples 50 --split train

HOW ARGPARSE WORKS:
    argparse lets you define named arguments that users pass from the
    command line. Each argument has:
        - a name (--dataset)
        - a type (str, int, etc.)
        - a default value (optional)
        - a help string (shown when user runs --help)
        - required=True/False
        - choices=[...] to restrict valid values
    
    When you run: python convert_to_schema.py --help
    It prints all available arguments automatically.
"""

import argparse
import json
import os
from datasets import load_dataset, load_from_disk
from typing import Dict, List, Optional


# ============================================================
# STEP 1: ARGUMENT PARSING
# ============================================================
# This is the "front door" of your script. It defines what
# the user can control without editing the code.

def parse_args():
    """Define and parse command-line arguments."""
    
    parser = argparse.ArgumentParser(
        description="Convert datasets to unified provenance-aware schema."
    )
    
    # Required argument: which dataset to convert
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["hotpotqa", "fever", "musique"],
        help="Which dataset to convert (hotpotqa, fever, or musique)"
    )
    
    # Required argument: where to save
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for the JSON file (e.g., data/hotpotqa_schema.json)"
    )
    
    # Optional: limit number of examples (useful for debugging)
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Max examples to convert (default: all). Use 50-100 for debugging."
    )
    
    # Optional: which split to use
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)"
    )
    
    # Optional: pretty-print the JSON
    parser.add_argument(
        "--pretty",
        action="store_true",  # This means: if --pretty is passed, it's True
        help="Pretty-print the output JSON (larger file but readable)"
    )
    
    # Optional: path to FEVER wiki lookup (needed for FEVER only)
    parser.add_argument(
        "--fever_wiki_lookup",
        type=str,
        default="data/fever_wiki_lookup.json",
        help="Path to FEVER wiki sentence lookup JSON (build with fever_wiki_resolver.py)"
    )

    # Optional: directory holding the MuSiQue dataset (needed for MuSiQue only).
    # MuSiQue is loaded from a local save_to_disk directory rather than the Hub,
    # so this points at the folder that contains one sub-directory per split
    # (e.g. <musique_dir>/train, <musique_dir>/validation). Relative to the repo root.
    parser.add_argument(
        "--musique_dir",
        type=str,
        default="data/raw/musique_rebuilt",
        help="Directory holding the MuSiQue dataset saved via save_to_disk, with one "
             "sub-directory per split (e.g. data/raw/musique_rebuilt/train). Adjust to "
             "wherever your MuSiQue copy lives."
    )
    
    return parser.parse_args()


# ============================================================
# STEP 2: HELPER FUNCTIONS
# ============================================================
# These create the individual pieces of your schema.
# By making them separate functions, you can reuse them
# across datasets.

def make_evidence_unit(
    evidence_id: str,
    text: str,
    doc_title: str,
    source_doc_id: str,
    sentence_index: Optional[int] = None,
    paragraph_index: Optional[int] = None,
    canonical_unit_type: str = "sentence",
    native_unit_type: str = "sentence",
    is_gold_evidence: bool = False,
    support_role: str = "unknown",
    native_label: Optional[str] = None,
    label_strength: str = "gold_sentence",
    supervision_weight: float = 1.0,
    text_status: str = "available",
    dataset: str = "hotpotqa",
) -> Dict:
    """
    Create a single evidence unit following the unified schema.
    
    WHY A HELPER FUNCTION?
    Each dataset formats evidence differently. By having one
    function that produces the standard schema, you only need
    to figure out the mapping once per dataset. The output
    format is always consistent.
    """
    return {
        "evidence_id": evidence_id,
        "text": text,
        "doc_title": doc_title,
        "source_doc_id": source_doc_id,
        
        "canonical_unit_type": canonical_unit_type,
        "native_unit_type": native_unit_type,
        
        "paragraph_index": paragraph_index,
        "sentence_index": sentence_index,
        "text_status": text_status,
        
        "is_gold_evidence": is_gold_evidence,
        "support_role": support_role,
        "native_label": native_label,
        "label_strength": label_strength,
        "supervision_weight": supervision_weight,
        
        "provenance": {
            "dataset": dataset,
            "source_type": "dataset_context",
            "doc_title": doc_title,
            "paragraph_index": paragraph_index,
            "sentence_index": sentence_index,
            "source_url": None,
            "timestamp": None,
            "version": None,
            "provenance_granularity": "sentence" if canonical_unit_type == "sentence" else "paragraph",
        }
    }


def make_schema_instance(
    example_id: str,
    dataset: str,
    task_type: str,
    input_text: str,
    input_type: str,
    evidence_units: List[Dict],
    evidence_state_label: str = "sufficient",
    target_answer: Optional[str] = None,
    target_label: Optional[str] = None,
    condition: str = "gold_full",
    agent_action_label: str = "answer",
) -> Dict:
    """
    Create a full schema instance from its components.
    
    This is the top-level structure that wraps everything together.
    """
    
    # Count gold evidence units
    num_gold = sum(1 for eu in evidence_units if eu["is_gold_evidence"])
    num_sources = len(set(eu["doc_title"] for eu in evidence_units))
    
    return {
        "example_id": example_id,
        "dataset": dataset,
        "task_type": task_type,
        "input_text": input_text,
        "input_type": input_type,
        
        "target_answer": target_answer,
        "target_label": target_label,
        
        "evidence_state_label": evidence_state_label,
        
        "evidence_set": {
            "condition": condition,
            "created_by": "dataset",
            "canonical_granularity": "sentence",
            "native_granularity": "sentence",
            "num_evidence_units": len(evidence_units),
            "num_gold_evidence_units": num_gold,
            "num_sources": num_sources,
        },
        
        "evidence_units": evidence_units,
        
        "agent_action_label": agent_action_label,
    }


# ============================================================
# STEP 3: DATASET-SPECIFIC CONVERSION
# ============================================================
# Each dataset has its own structure. We write one function
# per dataset that maps the raw format to our schema.

def convert_hotpotqa(raw_example: Dict, idx: int) -> Optional[Dict]:
    """
    Convert a single HotpotQA example to the unified schema.
    
    HotpotQA structure:
        - question: str
        - answer: str
        - supporting_facts: {
            "title": [list of doc titles],
            "sent_id": [list of sentence indices]
          }
        - context: {
            "title": [list of doc titles],
            "sentences": [list of list of sentences]
          }
        - type: "bridge" or "comparison"
        - level: "easy", "medium", "hard"
    
    The key challenge is matching supporting_facts to context.
    supporting_facts tells you WHICH sentences are gold evidence
    (by title + sentence index). context gives you the actual text.
    """
    
    question = raw_example["question"]
    answer = raw_example["answer"]
    
    # Build a lookup: (title, sent_idx) -> is_gold
    # This tells us which specific sentences are supporting facts
    gold_set = set()
    sf_titles = raw_example["supporting_facts"]["title"]
    sf_sent_ids = raw_example["supporting_facts"]["sent_id"]
    for title, sent_id in zip(sf_titles, sf_sent_ids):
        gold_set.add((title, sent_id))
    
    # Now iterate through all context and create evidence units
    evidence_units = []
    ctx_titles = raw_example["context"]["title"]
    ctx_sentences = raw_example["context"]["sentences"]
    
    for doc_idx, (title, sentences) in enumerate(zip(ctx_titles, ctx_sentences)):
        for sent_idx, sentence_text in enumerate(sentences):
            # Skip empty sentences
            if not sentence_text.strip():
                continue
            
            # Check if this sentence is a supporting fact
            is_gold = (title, sent_idx) in gold_set
            
            # Create a unique evidence ID
            # Format: hotpotqa_{example_idx}_{title_slug}_{sent_idx}
            title_slug = title.replace(" ", "_").replace("'", "")[:30]
            ev_id = f"hotpotqa_{idx:06d}_{title_slug}_sent_{sent_idx}"
            
            evidence_units.append(make_evidence_unit(
                evidence_id=ev_id,
                text=sentence_text,
                doc_title=title,
                source_doc_id=f"hotpotqa::{title}",
                sentence_index=sent_idx,
                paragraph_index=None,
                canonical_unit_type="sentence",
                native_unit_type="sentence",
                is_gold_evidence=is_gold,
                support_role="supports" if is_gold else "unknown",
                native_label="supporting_fact" if is_gold else None,
                label_strength="gold_sentence" if is_gold else "none",
                supervision_weight=1.0 if is_gold else 0.0,
                dataset="hotpotqa",
            ))
    
    # Only include examples that have at least one gold evidence
    num_gold = sum(1 for eu in evidence_units if eu["is_gold_evidence"])
    if num_gold == 0:
        return None
    
    # Create the full schema instance
    example_id = f"hotpotqa_{idx:06d}_gold_full"
    
    return make_schema_instance(
        example_id=example_id,
        dataset="hotpotqa",
        task_type="multi_hop_qa",
        input_text=question,
        input_type="question",
        evidence_units=evidence_units,
        evidence_state_label="sufficient",
        target_answer=answer,
        target_label=None,
        condition="gold_full",
        agent_action_label="answer",
    )

def convert_fever(raw_example: Dict, idx: int, wiki_lookup: Optional[Dict] = None) -> Optional[Dict]:
    """
    Convert a single FEVER example to the unified schema.
    
    FEVER structure (HuggingFace v1.0):
        - claim: str
        - label: "SUPPORTS", "REFUTES", or "NOT ENOUGH INFO"
        - evidence_annotation_id: int (single value per row)
        - evidence_id: int (single value per row)
        - evidence_wiki_url: str (single value per row)
        - evidence_sentence_id: int (single value per row)
    
    Note: HuggingFace stores one claim-evidence pair per row.
    Multiple evidence items for the same claim appear as separate rows.
    
    KEY DIFFERENCES FROM HOTPOTQA:
    1. Input is a "claim" not a "question"
    2. Label is a verification label, not an answer
    3. Three classes map naturally:
       SUPPORTS    → sufficient (gold evidence supports the claim)
       REFUTES     → contradicted (evidence contradicts the claim)
       NOT ENOUGH INFO → insufficient (not enough evidence)
    """
    
    claim = raw_example["claim"]
    label = raw_example["label"]
    
    # Map FEVER labels to evidence states
    label_to_state = {
        "SUPPORTS": "sufficient",
        "REFUTES": "contradicted",
        "NOT ENOUGH INFO": "insufficient",
    }
    label_to_action = {
        "SUPPORTS": "answer",
        "REFUTES": "flag_conflict",
        "NOT ENOUGH INFO": "retrieve_more",
    }
    label_to_role = {
        "SUPPORTS": "supports",
        "REFUTES": "refutes",
        "NOT ENOUGH INFO": "unknown",
    }
    
    evidence_state = label_to_state.get(label, "insufficient")
    agent_action = label_to_action.get(label, "retrieve_more")
    support_role = label_to_role.get(label, "unknown")
    
    # Extract evidence
    # HuggingFace FEVER v1.0 stores evidence as SINGLE values per row,
    # not as lists. Handle both cases for safety.
    evidence_units = []
    
    wiki_url = raw_example.get("evidence_wiki_url")
    sent_id = raw_example.get("evidence_sentence_id")
    
    # Normalize: if they're lists, take all; if single values, wrap in list
    if isinstance(wiki_url, list):
        wiki_urls = wiki_url
        sent_ids = raw_example.get("evidence_sentence_id", [])
        if not isinstance(sent_ids, list):
            sent_ids = [sent_ids]
    else:
        wiki_urls = [wiki_url] if wiki_url else []
        sent_ids = [sent_id] if sent_id is not None else []
    
    # Process each evidence reference
    seen_evidence = set()
    for ev_idx in range(len(wiki_urls)):
        w_url = wiki_urls[ev_idx] if ev_idx < len(wiki_urls) else None
        s_id = sent_ids[ev_idx] if ev_idx < len(sent_ids) else None
        
        # Skip empty or None
        if not w_url or w_url == "":
            continue
        
        # Skip NOT ENOUGH INFO (they have no real evidence)
        if label == "NOT ENOUGH INFO":
            continue
        
        # Deduplicate
        ev_key = (w_url, s_id)
        if ev_key in seen_evidence:
            continue
        seen_evidence.add(ev_key)
        
        # Resolve text from wiki lookup
        resolved_text = None
        text_status = "unresolved_reference"
        
        if wiki_lookup and s_id is not None:
            page_data = (wiki_lookup.get(str(w_url)) or 
                        wiki_lookup.get(str(w_url).replace(" ", "_")) or
                        wiki_lookup.get(str(w_url).replace("_", " ")))
            if page_data:
                resolved_text = page_data.get(int(s_id))
                if resolved_text:
                    text_status = "available"
        
        page_slug = str(w_url).replace(" ", "_")[:30]
        ev_id = f"fever_{idx:06d}_{page_slug}_sent_{s_id}"
        
        evidence_units.append(make_evidence_unit(
            evidence_id=ev_id,
            text=resolved_text,
            doc_title=str(w_url),
            source_doc_id=f"fever::{w_url}",
            sentence_index=int(s_id) if s_id is not None else None,
            canonical_unit_type="sentence",
            native_unit_type="sentence",
            is_gold_evidence=True,
            support_role=support_role,
            native_label=label,
            label_strength="gold_sentence",
            supervision_weight=1.0,
            text_status=text_status,
            dataset="fever",
        ))
    
    # For NOT ENOUGH INFO, create a placeholder
    if label == "NOT ENOUGH INFO" and len(evidence_units) == 0:
        evidence_units.append(make_evidence_unit(
            evidence_id=f"fever_{idx:06d}_no_evidence",
            text=None,
            doc_title="none",
            source_doc_id="fever::none",
            is_gold_evidence=False,
            support_role="unknown",
            native_label=label,
            label_strength="none",
            supervision_weight=0.0,
            text_status="no_evidence",
            dataset="fever",
        ))
    
    example_id = f"fever_{idx:06d}_{evidence_state}"
    
    return make_schema_instance(
        example_id=example_id,
        dataset="fever",
        task_type="claim_verification",
        input_text=claim,
        input_type="claim",
        evidence_units=evidence_units,
        evidence_state_label=evidence_state,
        target_answer=None,
        target_label=label,
        condition="gold_full" if label == "SUPPORTS" else (
            "refuted" if label == "REFUTES" else "no_evidence"
        ),
        agent_action_label=agent_action,
    )

def convert_musique(raw_example: Dict, idx: int) -> Optional[Dict]:
    """
    Convert a single MuSiQue example to the unified schema.
    
    MuSiQue structure:
        - question: str
        - answer: str
        - paragraphs: list of dicts, each with:
            - title: str
            - paragraph_text: str
            - is_supporting: bool
            - idx: int
        - question_decomposition: list of sub-questions (optional)
        - answerable: bool
    
    KEY DIFFERENCES FROM HOTPOTQA:
    1. Evidence is at paragraph level, not sentence level
    2. Each paragraph explicitly has is_supporting: true/false
       (much simpler than HotpotQA's matching logic!)
    3. Has multi-hop decomposition info we can use for
       hop_completeness later
    4. Has an "answerable" field — unanswerable examples
       are natural "insufficient" variants
    """
    
    question = raw_example["question"]
    answer = raw_example.get("answer", "")
    answerable = raw_example.get("answerable", True)
    paragraphs = raw_example.get("paragraphs", [])
    
    evidence_units = []
    
    for para in paragraphs:
        title = para.get("title", "unknown")
        text = para.get("paragraph_text", "")
        is_supporting = para.get("is_supporting", False)
        para_idx = para.get("idx", 0)
        
        if not text.strip():
            continue
        
        title_slug = title.replace(" ", "_").replace("'", "")[:30]
        ev_id = f"musique_{idx:06d}_{title_slug}_para_{para_idx}"
        
        evidence_units.append(make_evidence_unit(
            evidence_id=ev_id,
            text=text,
            doc_title=title,
            source_doc_id=f"musique::{title}",
            paragraph_index=para_idx,
            sentence_index=None,
            canonical_unit_type="paragraph",
            native_unit_type="paragraph",
            is_gold_evidence=is_supporting,
            support_role="supports" if is_supporting else "unknown",
            native_label="supporting_paragraph" if is_supporting else None,
            label_strength="gold_paragraph" if is_supporting else "none",
            # Paragraph-level labels get slightly lower weight
            # than sentence-level gold labels
            supervision_weight=0.8 if is_supporting else 0.0,
            dataset="musique",
        ))
    
    # Determine evidence state based on answerability
    if answerable:
        evidence_state = "sufficient"
        agent_action = "answer"
        condition = "gold_full"
    else:
        evidence_state = "insufficient"
        agent_action = "retrieve_more"
        condition = "unanswerable"
    
    num_gold = sum(1 for eu in evidence_units if eu["is_gold_evidence"])
    if num_gold == 0 and answerable:
        return None  # Skip examples with no supporting paragraphs
    
    example_id = f"musique_{idx:06d}_{evidence_state}"
    
    return make_schema_instance(
        example_id=example_id,
        dataset="musique",
        task_type="decomposed_multi_hop_qa",
        input_text=question,
        input_type="question",
        evidence_units=evidence_units,
        evidence_state_label=evidence_state,
        target_answer=answer if answerable else None,
        target_label=None,
        condition=condition,
        agent_action_label=agent_action,
    )


# ============================================================
# STEP 4: MAIN CONVERSION LOOP
# ============================================================
# This ties everything together: load the dataset, convert
# each example, collect results, and save.

def run_conversion(args):
    """Main conversion pipeline."""
    
    print(f"{'='*60}")
    print(f"Converting {args.dataset} to unified schema")
    print(f"{'='*60}")
    
    # --- Load dataset ---
    print(f"\n[1/4] Loading {args.dataset} ({args.split} split)...")
    
    if args.dataset == "hotpotqa":
        # HotpotQA has two configs: "fullwiki" and "distractor"
        # "distractor" is more common for QA research.
        # Use the namespaced Hub id ("hotpotqa/hotpot_qa"); the bare
        # "hotpot_qa" alias is deprecated on recent `datasets` versions.
        raw_dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split=args.split)
        convert_fn = convert_hotpotqa
        
    elif args.dataset == "fever":
        raw_dataset = load_dataset("fever", "v1.0", split=args.split)
        
        # Load wiki lookup to resolve evidence text
        wiki_lookup = None
        if os.path.exists(args.fever_wiki_lookup):
            print(f"  Loading FEVER wiki lookup from {args.fever_wiki_lookup}...")
            with open(args.fever_wiki_lookup, "r", encoding="utf-8") as f:
                raw_lookup = json.load(f)
            # Convert string sentence IDs back to int
            wiki_lookup = {}
            for page_id, sentences in raw_lookup.items():
                wiki_lookup[page_id] = {int(k): v for k, v in sentences.items()}
            print(f"  Loaded {len(wiki_lookup)} wiki pages")
        else:
            print(f"  WARNING: Wiki lookup not found at {args.fever_wiki_lookup}")
            print(f"  Evidence text will be unresolved. Run fever_wiki_resolver.py --build first.")
        
        # Use functools.partial to pass wiki_lookup to convert_fever
        from functools import partial
        convert_fn = partial(convert_fever, wiki_lookup=wiki_lookup)
        
    elif args.dataset == "musique":
        # MuSiQue is loaded from a local save_to_disk directory (one sub-dir per split),
        # not from the Hub. The directory is configurable via --musique_dir and is
        # relative to the repo root (default: data/raw/musique_rebuilt).
        # NOTE: download_dataset.py writes MuSiQue to data/raw/musique; if your copy
        # lives there (or anywhere else), pass --musique_dir to match.
        split_path = os.path.join(args.musique_dir, args.split)
        if not os.path.isdir(split_path):
            raise FileNotFoundError(
                f"MuSiQue split not found at '{split_path}'. "
                f"Point --musique_dir at the directory that holds the MuSiQue splits "
                f"saved via Dataset.save_to_disk / load_dataset(...).save_to_disk "
                f"(expected a sub-directory named '{args.split}' inside it)."
            )
        raw_dataset = load_from_disk(split_path)
        convert_fn = convert_musique
        
    print(f"  Loaded {len(raw_dataset)} examples")
    
    # --- Apply max_examples limit ---
    if args.max_examples:
        raw_dataset = raw_dataset.select(range(min(args.max_examples, len(raw_dataset))))
        print(f"  Limited to {len(raw_dataset)} examples (--max_examples)")
    
    # --- Convert each example ---
    print(f"\n[2/4] Converting examples...")
    converted = []
    skipped = 0
    
    for idx, raw_example in enumerate(raw_dataset):
        result = convert_fn(raw_example, idx)
        
        if result is not None:
            converted.append(result)
        else:
            skipped += 1
        
        # Progress update every 1000 examples
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(raw_dataset)} "
                  f"(converted: {len(converted)}, skipped: {skipped})")
    
    print(f"  Done: {len(converted)} converted, {skipped} skipped")
    
    # --- Print stats ---
    print(f"\n[3/4] Schema statistics:")
    total_evidence_units = sum(
        len(inst["evidence_units"]) for inst in converted
    )
    total_gold_units = sum(
        inst["evidence_set"]["num_gold_evidence_units"] for inst in converted
    )
    avg_evidence = total_evidence_units / len(converted) if converted else 0
    avg_gold = total_gold_units / len(converted) if converted else 0
    
    print(f"  Total instances: {len(converted)}")
    print(f"  Total evidence units: {total_evidence_units}")
    print(f"  Total gold evidence units: {total_gold_units}")
    print(f"  Avg evidence units per instance: {avg_evidence:.1f}")
    print(f"  Avg gold evidence per instance: {avg_gold:.1f}")
    
    # Verify a sample
    if converted:
        sample = converted[0]
        print(f"\n  Sample verification (first instance):")
        print(f"    ID: {sample['example_id']}")
        print(f"    Question: {sample['input_text'][:80]}...")
        print(f"    Answer: {sample['target_answer']}")
        print(f"    Evidence units: {sample['evidence_set']['num_evidence_units']}")
        print(f"    Gold units: {sample['evidence_set']['num_gold_evidence_units']}")
        print(f"    Sources: {sample['evidence_set']['num_sources']}")
    
    # --- Save ---
    print(f"\n[4/4] Saving to {args.output}...")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    
    with open(args.output, "w", encoding="utf-8") as f:
        if args.pretty:
            json.dump(converted, f, indent=2, ensure_ascii=False)
        else:
            json.dump(converted, f, ensure_ascii=False)
    
    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"  Saved {len(converted)} instances ({file_size_mb:.1f} MB)")
    print(f"\n{'='*60}")
    print(f"DONE — {args.dataset} conversion complete!")
    print(f"{'='*60}")


# ============================================================
# STEP 5: ENTRY POINT
# ============================================================
# This is the standard Python pattern for "run this when
# the script is executed directly, but not when imported."

if __name__ == "__main__":
    args = parse_args()
    run_conversion(args)
