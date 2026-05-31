"""
preprocess_light.py
====================
Convert the heavy unified JSON files (~3.2 GB) into lightweight 
versions (~200-400 MB) that load in seconds instead of 45 minutes.

Run ONCE, then use the _light.json files for all training experiments.

Usage:
    python preprocess_light.py --data_dir data/ --output_dir data/
"""

import json
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Create lightweight training files")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing unified_train.json and unified_val.json")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (defaults to data_dir)")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or args.data_dir
    os.makedirs(output_dir, exist_ok=True)
    
    for split in ["train", "val"]:
        input_path = os.path.join(args.data_dir, f"unified_{split}.json")
        output_path = os.path.join(output_dir, f"unified_{split}_light.json")
        
        if not os.path.exists(input_path):
            print(f"  Skipping {split} — {input_path} not found")
            continue
        
        input_size = os.path.getsize(input_path) / (1024 * 1024)
        print(f"\nProcessing {split} ({input_size:.1f} MB)...")
        print(f"  Loading {input_path} (this may take a few minutes for train)...")
        
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        print(f"  Loaded {len(data)} instances")
        print(f"  Extracting text + labels + features...")
        
        light = []
        for i, inst in enumerate(data):
            # Pre-extract and concatenate evidence text
            evidence_texts = []
            for eu in inst.get("evidence_units", []):
                if eu.get("text") and eu["text"].strip():
                    evidence_texts.append(eu["text"].strip())
            
            evidence_text = " ".join(evidence_texts)
            # Cap at 3000 chars to save space (DeBERTa truncates to 512 tokens anyway)
            if len(evidence_text) > 3000:
                evidence_text = evidence_text[:3000]
            
            light.append({
                "input_text": inst["input_text"],
                "evidence_text": evidence_text,
                "evidence_state_label": inst["evidence_state_label"],
                "graph_features": inst.get("graph_features", {}),
                "dataset": inst.get("dataset", "unknown"),
                "example_id": inst.get("example_id", f"{split}_{i}"),
            })
            
            if (i + 1) % 50000 == 0:
                print(f"    Processed {i + 1}/{len(data)}")
        
        print(f"  Saving to {output_path}...")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(light, f, ensure_ascii=False)
        
        output_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  Saved ({output_size:.1f} MB) — {input_size/output_size:.1f}x smaller")
        
        # Free memory
        del data, light
    
    print(f"\nDone! Use the _light.json files for training:")
    print(f"  python train_detector.py --train_path {os.path.join(output_dir, 'unified_train_light.json')} --val_path {os.path.join(output_dir, 'unified_val_light.json')} ...")


if __name__ == "__main__":
    main()