"""
download_models_local.py
========================
Download models to a local directory, then transfer to remote server.

Usage:
    python download_models_local.py --output_dir ./models

Then transfer to server:
    scp -r ./models user@server:/path/to/project/models/

Then in train_detector.py, use local path instead of HuggingFace name:
    --model_path /path/to/project/models/deberta-v3-base
"""

import os
import argparse
from transformers import AutoModel, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./models",
                        help="Local directory to save models")
    return parser.parse_args()


def main():
    args = parse_args()
    
    models = [
        ("microsoft/deberta-v3-base", "deberta-v3-base", True),
        ("bert-base-uncased", "bert-base-uncased", False),
        ("roberta-base", "roberta-base", False),
    ]
    
    for model_name, folder_name, slow_tokenizer in models:
        save_path = os.path.join(args.output_dir, folder_name)
        os.makedirs(save_path, exist_ok=True)
        
        print(f"\n{'='*50}")
        print(f"Downloading: {model_name}")
        print(f"Saving to: {save_path}")
        print(f"{'='*50}")
        
        try:
            print(f"  Tokenizer...")
            if slow_tokenizer:
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.save_pretrained(save_path)
            
            print(f"  Model...")
            model = AutoModel.from_pretrained(model_name)
            model.save_pretrained(save_path)
            
            print(f"  Saved to {save_path}")
        except Exception as e:
            print(f"  ERROR: {e}")
    
    print(f"\n{'='*50}")
    print(f"All models saved to {args.output_dir}/")
    print(f"\nTransfer to server:")
    print(f"  scp -r {args.output_dir} user@server:/path/to/project/")
    print(f"\nThen use local paths in train_detector.py")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()