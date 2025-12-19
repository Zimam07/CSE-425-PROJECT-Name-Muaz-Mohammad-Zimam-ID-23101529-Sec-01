"""
Orchestrator: create demo dataset and run the full pipeline end-to-end.

Usage:
    python scripts/run_demo_pipeline.py

This will:
 - Create demo data (data/raw)
 - Validate and align metadata
 - Extract audio features
 - Generate lyrics embeddings (fallback to random if model unavailable)
 - Prepare multimodal features
 - Run quick evaluation (PCA + KMeans) and print/save metrics
"""
import subprocess
import sys
import os

steps = [
    ['python', 'scripts/create_demo_dataset.py', '--n', '12'],
    ['python', 'scripts/validate_dataset.py', '--audio_dir', 'data/raw/audio', '--lyrics_dir', 'data/raw/lyrics', '--metadata', 'data/raw/metadata.csv', '--out', 'data/raw/metadata_aligned.csv'],
    ['python', 'src/features.py', '--input_dir', 'data/raw/audio', '--output_dir', 'data/features', '--ext', '.wav'],
    ['python', 'scripts/lyrics_to_embeddings.py', '--metadata', 'data/raw/metadata_aligned.csv', '--out_dir', 'data/features/lyrics_embeddings', '--model', 'all-MiniLM-L6-v2'],
    ['python', 'src/prepare_multimodal.py', '--audio_dir', 'data/features', '--lyrics_dir', 'data/features/lyrics_embeddings', '--out_dir', 'data/features/multimodal'],
    ['python', 'scripts/quick_eval.py', '--feat_dir', 'data/features/multimodal', '--k', '2', '--out', 'results/demo_metrics.csv']
]

for cmd in steps:
    print('Running:', ' '.join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print('Step failed:', ' '.join(cmd))
        sys.exit(proc.returncode)

print('Demo pipeline completed. See results/demo_metrics.csv')
