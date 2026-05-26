"""
download_fever_wiki.py
======================
Downloads the FEVER Wikipedia corpus and builds a sentence lookup.
Tries multiple sources automatically.

Usage:
    python download_fever_wiki.py --output data/fever_wiki_lookup.json
"""

import argparse
import json
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Download and build FEVER wiki lookup")
    parser.add_argument("--output", type=str, default="data/fever_wiki_lookup.json",
                        help="Output path for the lookup JSON")
    parser.add_argument("--method", type=str, default="auto",
                        choices=["auto", "hf_wiki", "hf_fever_nli", "download"],
                        help="Which download method to use")
    return parser.parse_args()


def try_hf_wiki_pages(output_path):
    """Method 1: Load wiki_pages config from HuggingFace FEVER dataset."""
    print("\n--- Method 1: HuggingFace fever/wiki_pages ---")
    try:
        from datasets import load_dataset
        ds = load_dataset("fever", "wiki_pages", 
                         split="wikipedia_pages", 
                         trust_remote_code=True)
        print(f"  Loaded {len(ds)} pages")
        return build_lookup_from_hf(ds, output_path)
    except Exception as e:
        print(f"  Failed: {e}")
        return False


def try_hf_fever_nli(output_path):
    """Method 2: Use a pre-processed FEVER dataset that includes evidence text."""
    print("\n--- Method 2: HuggingFace pre-processed FEVER ---")
    
    # Try multiple HuggingFace sources that might have resolved evidence
    sources = [
        ("pietrolesci/fever", None),
        ("copenlu/fever_gold_evidence", None),
        ("fever", "v1.0"),
    ]
    
    for dataset_name, config in sources:
        try:
            from datasets import load_dataset
            print(f"  Trying {dataset_name} (config={config})...")
            if config:
                ds = load_dataset(dataset_name, config, split="train", 
                                 trust_remote_code=True)
            else:
                ds = load_dataset(dataset_name, split="train",
                                 trust_remote_code=True)
            
            print(f"  Loaded {len(ds)} examples")
            print(f"  Columns: {ds.column_names}")
            print(f"  First example keys: {list(ds[0].keys())}")
            
            # Check if evidence text is available
            first = ds[0]
            print(f"  Sample: {json.dumps({k: str(v)[:100] for k, v in first.items()}, indent=2)}")
            
            return ds  # Return the dataset for inspection
            
        except Exception as e:
            print(f"  Failed: {e}")
            continue
    
    return None


def try_download_wiki_zip(output_path):
    """Method 3: Download wiki-pages.zip from FEVER website."""
    print("\n--- Method 3: Direct download from FEVER website ---")
    
    import urllib.request
    import zipfile
    import tempfile
    
    urls = [
        "https://s3-eu-west-1.amazonaws.com/fever.public/wiki-pages.zip",
        "https://fever.ai/download/fever/wiki-pages.zip",
    ]
    
    zip_path = os.path.join(tempfile.gettempdir(), "fever-wiki-pages.zip")
    extract_dir = os.path.join(tempfile.gettempdir(), "fever-wiki")
    
    for url in urls:
        try:
            print(f"  Downloading from {url}...")
            print(f"  This is ~1.6 GB, may take 5-15 minutes...")
            
            # Download with progress
            def progress_hook(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    pct = min(100, downloaded * 100 / total_size)
                    mb = downloaded / (1024 * 1024)
                    total_mb = total_size / (1024 * 1024)
                    sys.stdout.write(f"\r  {mb:.0f}/{total_mb:.0f} MB ({pct:.1f}%)")
                    sys.stdout.flush()
            
            urllib.request.urlretrieve(url, zip_path, progress_hook)
            print(f"\n  Downloaded to {zip_path}")
            
            # Unzip
            print(f"  Extracting...")
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(extract_dir)
            print(f"  Extracted to {extract_dir}")
            
            # Build lookup from extracted files
            return build_lookup_from_jsonl(extract_dir, output_path)
            
        except Exception as e:
            print(f"  Failed: {e}")
            continue
    
    return False


def build_lookup_from_hf(ds, output_path):
    """Build lookup from a HuggingFace wiki_pages dataset."""
    lookup = {}
    total_sentences = 0
    
    for i, page in enumerate(ds):
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
                        total_sentences += 1
                except (ValueError, IndexError):
                    continue
        
        if sentences:
            lookup[page_id] = sentences
        
        if (i + 1) % 500000 == 0:
            print(f"  Processed {i + 1} pages ({len(lookup)} with sentences)")
    
    print(f"  Total: {len(lookup)} pages, {total_sentences} sentences")
    return save_lookup(lookup, output_path)


def build_lookup_from_jsonl(wiki_dir, output_path):
    """Build lookup from extracted wiki-pages JSONL files."""
    from pathlib import Path
    
    # Find JSONL files (might be in a subdirectory)
    wiki_path = Path(wiki_dir)
    jsonl_files = sorted(wiki_path.rglob("wiki-*.jsonl"))
    
    if not jsonl_files:
        print(f"  No wiki-*.jsonl files found in {wiki_dir}")
        return False
    
    print(f"  Found {len(jsonl_files)} JSONL files")
    
    lookup = {}
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
        
        if (file_idx + 1) % 5 == 0:
            print(f"  Processed {file_idx + 1}/{len(jsonl_files)} files "
                  f"({len(lookup)} pages)")
    
    print(f"  Total: {len(lookup)} pages, {total_sentences} sentences")
    return save_lookup(lookup, output_path)


def save_lookup(lookup, output_path):
    """Save the lookup dict to JSON."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    # Convert int keys to strings for JSON
    json_lookup = {}
    for page_id, sentences in lookup.items():
        json_lookup[page_id] = {str(k): v for k, v in sentences.items()}
    
    print(f"  Saving to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(json_lookup, f, ensure_ascii=False)
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Saved ({file_size:.1f} MB)")
    return True


def main():
    args = parse_args()
    
    print("=" * 60)
    print("FEVER Wikipedia Corpus Download & Build")
    print("=" * 60)
    
    if args.method == "auto":
        # Try methods in order of speed
        if try_hf_wiki_pages(args.output):
            print("\\nSuccess with HuggingFace wiki_pages!")
            return
        
        print("\nMethod 1 failed, trying Method 3 (direct download)...")
        if try_download_wiki_zip(args.output):
            print("\nSuccess with direct download!")
            return
        
        print("\nAll methods failed.")
        print("Please download manually from https://fever.ai/resources.html")
        print("Then run: python fever_wiki_resolver.py --build --wiki_dir <path>")
    
    elif args.method == "hf_wiki":
        try_hf_wiki_pages(args.output)
    
    elif args.method == "hf_fever_nli":
        result = try_hf_fever_nli(args.output)
        if result is not None:
            print("\nDataset loaded. Check the output above to see if evidence text is available.")
    
    elif args.method == "download":
        try_download_wiki_zip(args.output)


if __name__ == "__main__":
    main()