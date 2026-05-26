"""
fever_wiki_resolver.py
======================
Downloads the FEVER Wikipedia corpus and builds a lookup
so you can resolve evidence references to actual text.

FEVER stores evidence as (wiki_page, sentence_id) but doesn't
include the sentence text. The wiki corpus provides the text.

Usage:
    # Step 1: Download and build the lookup (do this once)
    python fever_wiki_resolver.py --build --wiki_dir data/fever_wiki --output data/fever_wiki_lookup.json

    # Step 2: Use the lookup in your conversion script
    # (see resolve_fever_evidence() function below)

The wiki corpus comes from: https://fever.ai/resources.html
It's also available via HuggingFace in some dataset versions.
"""

import argparse
import json
import os
import zipfile
from pathlib import Path
from typing import Dict, Optional


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build FEVER wiki sentence lookup"
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build the wiki lookup from raw wiki-pages files"
    )
    parser.add_argument(
        "--wiki_dir",
        type=str,
        default="data/fever_wiki",
        help="Directory containing wiki-pages JSONL files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/fever_wiki_lookup.json",
        help="Output path for the lookup JSON"
    )
    parser.add_argument(
        "--test_page",
        type=str,
        default=None,
        help="Test lookup with a specific page title"
    )
    parser.add_argument(
        "--test_sent_id",
        type=int,
        default=0,
        help="Test lookup with a specific sentence ID"
    )
    return parser.parse_args()


def download_fever_wiki():
    """
    Instructions for getting the FEVER wiki corpus.
    
    The corpus is ~1.6GB compressed. You have two options:
    
    OPTION A (recommended): Download from FEVER website
        1. Go to https://fever.ai/resources.html
        2. Download "Pre-processed Wikipedia Pages" (wiki-pages.zip)
        3. Unzip into data/fever_wiki/
        4. You should see files like wiki-001.jsonl, wiki-002.jsonl, etc.
    
    OPTION B: Use the HuggingFace version
        Some HuggingFace FEVER datasets include resolved text.
        Try: datasets.load_dataset("fever", "wiki_pages")
        or:  datasets.load_dataset("pietrolesci/fever_wikipedia")
    """
    print("=" * 60)
    print("FEVER Wiki Corpus Download Instructions")
    print("=" * 60)
    print()
    print("The FEVER wiki corpus is needed to resolve evidence text.")
    print()
    print("Option A: Manual download")
    print("  1. Visit: https://fever.ai/resources.html")
    print("  2. Download 'Pre-processed Wikipedia Pages' (~1.6GB)")
    print("  3. Unzip to: data/fever_wiki/")
    print("  4. Files should look like: wiki-001.jsonl, wiki-002.jsonl, ...")
    print()
    print("Option B: Try HuggingFace (may be faster)")
    print("  python -c \"from datasets import load_dataset; ds = load_dataset('pietrolesci/fever_wikipedia')\"")
    print()
    print("Once downloaded, run:")
    print("  python fever_wiki_resolver.py --build --wiki_dir data/fever_wiki")
    print()


def build_lookup_from_jsonl(wiki_dir: str, output_path: str):
    """
    Build a lookup dict from FEVER wiki-pages JSONL files.
    
    Each line in the JSONL files looks like:
    {
        "id": "Page_Title",
        "text": "",
        "lines": "0\tFirst sentence\n1\tSecond sentence\n..."
    }
    
    The "lines" field contains tab-separated (index, sentence) pairs,
    separated by newlines.
    
    We build: lookup[page_title][sentence_id] = sentence_text
    """
    wiki_dir = Path(wiki_dir)
    
    if not wiki_dir.exists():
        print(f"Error: {wiki_dir} does not exist.")
        download_fever_wiki()
        return
    
    # Find all JSONL files
    jsonl_files = sorted(wiki_dir.glob("wiki-*.jsonl"))
    if not jsonl_files:
        # Maybe they're in a subdirectory
        jsonl_files = sorted(wiki_dir.rglob("wiki-*.jsonl"))
    
    if not jsonl_files:
        print(f"Error: No wiki-*.jsonl files found in {wiki_dir}")
        download_fever_wiki()
        return
    
    print(f"Found {len(jsonl_files)} wiki JSONL files")
    
    lookup = {}
    total_pages = 0
    total_sentences = 0
    
    for file_idx, jsonl_file in enumerate(jsonl_files):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    page = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                page_id = page.get("id", "")
                lines_str = page.get("lines", "")
                
                if not page_id or not lines_str:
                    continue
                
                # Parse the "lines" field
                # Format: "0\tFirst sentence.\n1\tSecond sentence.\n..."
                sentences = {}
                for line_entry in lines_str.split("\n"):
                    parts = line_entry.split("\t")
                    if len(parts) >= 2:
                        try:
                            sent_id = int(parts[0])
                            sent_text = parts[1].strip()
                            if sent_text:
                                sentences[sent_id] = sent_text
                                total_sentences += 1
                        except (ValueError, IndexError):
                            continue
                
                if sentences:
                    lookup[page_id] = sentences
                    total_pages += 1
        
        if (file_idx + 1) % 10 == 0:
            print(f"  Processed {file_idx + 1}/{len(jsonl_files)} files "
                  f"({total_pages} pages, {total_sentences} sentences)")
    
    print(f"\nFinal: {total_pages} pages, {total_sentences} sentences")
    
    # Save lookup
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    print(f"Saving lookup to {output_path}...")
    
    # Convert int keys to strings for JSON serialization
    json_lookup = {}
    for page_id, sentences in lookup.items():
        json_lookup[page_id] = {str(k): v for k, v in sentences.items()}
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_lookup, f, ensure_ascii=False)
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Saved ({file_size_mb:.1f} MB)")
    
    return lookup


def build_lookup_from_huggingface(output_path: str):
    """
    Alternative: build lookup from a HuggingFace version of the 
    FEVER wiki corpus. This avoids manual download.
    
    Try this if you don't want to download the zip manually.
    """
    try:
        from datasets import load_dataset
        
        print("Loading FEVER Wikipedia from HuggingFace...")
        print("(This may take a while on first download)")
        
        # Try different HuggingFace sources
        try:
            wiki_ds = load_dataset("fever", "wiki_pages", split="wikipedia_pages")
        except Exception:
            try:
                wiki_ds = load_dataset("pietrolesci/fever_wikipedia", split="train")
            except Exception:
                print("Could not load FEVER wiki from HuggingFace.")
                print("Please download manually from https://fever.ai/resources.html")
                return None
        
        print(f"Loaded {len(wiki_ds)} pages")
        
        lookup = {}
        for page in wiki_ds:
            page_id = page.get("id", "")
            lines_str = page.get("lines", "")
            
            if not page_id or not lines_str:
                continue
            
            sentences = {}
            for line_entry in lines_str.split("\n"):
                parts = line_entry.split("\t")
                if len(parts) >= 2:
                    try:
                        sent_id = int(parts[0])
                        sent_text = parts[1].strip()
                        if sent_text:
                            sentences[sent_id] = sent_text
                    except (ValueError, IndexError):
                        continue
            
            if sentences:
                lookup[page_id] = sentences
        
        # Save
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        json_lookup = {
            pid: {str(k): v for k, v in sents.items()}
            for pid, sents in lookup.items()
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(json_lookup, f, ensure_ascii=False)
        
        print(f"Saved lookup with {len(lookup)} pages to {output_path}")
        return lookup
        
    except ImportError:
        print("datasets library not installed. Run: pip install datasets")
        return None


def load_lookup(lookup_path: str) -> Dict:
    """Load a previously built lookup from disk."""
    print(f"Loading wiki lookup from {lookup_path}...")
    with open(lookup_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    
    # Convert string keys back to int
    lookup = {}
    for page_id, sentences in raw.items():
        lookup[page_id] = {int(k): v for k, v in sentences.items()}
    
    print(f"  Loaded {len(lookup)} pages")
    return lookup


def resolve_evidence_text(
    wiki_url: str,
    sentence_id: int,
    lookup: Dict
) -> Optional[str]:
    """
    Resolve a FEVER evidence reference to actual text.
    
    Args:
        wiki_url: The Wikipedia page identifier (e.g., "Nikolaj_Coster-Waldau")
        sentence_id: The sentence index within that page
        lookup: The wiki lookup dictionary
    
    Returns:
        The sentence text, or None if not found
    """
    page = lookup.get(wiki_url) or lookup.get(wiki_url.replace("_", " "))
    if page is None:
        return None
    
    return page.get(sentence_id)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    args = parse_args()
    
    if args.build:
        wiki_dir = Path(args.wiki_dir)
        
        if wiki_dir.exists() and list(wiki_dir.rglob("wiki-*.jsonl")):
            print("Building lookup from local JSONL files...")
            build_lookup_from_jsonl(args.wiki_dir, args.output)
        else:
            print("No local wiki files found. Trying HuggingFace...")
            build_lookup_from_huggingface(args.output)
    
    elif args.test_page:
        if not os.path.exists(args.output):
            print(f"Lookup file not found: {args.output}")
            print("Run with --build first.")
        else:
            lookup = load_lookup(args.output)
            text = resolve_evidence_text(args.test_page, args.test_sent_id, lookup)
            if text:
                print(f"\nPage: {args.test_page}")
                print(f"Sentence {args.test_sent_id}: {text}")
            else:
                print(f"Could not find {args.test_page} sentence {args.test_sent_id}")
    
    else:
        download_fever_wiki()