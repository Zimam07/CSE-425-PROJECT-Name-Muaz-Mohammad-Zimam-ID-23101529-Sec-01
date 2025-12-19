"""
Helper script to download Jamendo dataset (or other Kaggle datasets).
Requires: Kaggle API configured (`kaggle.json` at %USERPROFILE%/.kaggle/kaggle.json)

Usage (example):
    python scripts/download_jamendo.py --dataset <dataset-slug> --out_dir data/raw

Note: Jamendo dataset on Kaggle may be under a specific user slug. If you prefer manual download, place files under `data/raw`.
"""
import argparse
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='', help='Kaggle dataset slug (e.g., owner/dataset-name)')
parser.add_argument('--out_dir', type=str, default='data/raw')
args = parser.parse_args()

if not args.dataset:
    print("Please supply a Kaggle dataset slug, or download manually and place files in data/raw.")
    print("If you have kaggle API configured, run: kaggle datasets download -d <owner/dataset> -p data/raw")
    exit(1)

os.makedirs(args.out_dir, exist_ok=True)
print(f"Downloading {args.dataset} to {args.out_dir}. Ensure kaggle is configured.")
cmd = ["kaggle", "datasets", "download", "-d", args.dataset, "-p", args.out_dir, "--unzip"]
subprocess.run(cmd, check=True)
print("Done.")
