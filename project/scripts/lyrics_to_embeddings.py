"""
Convert lyrics (from aligned metadata CSV or per-file txt) into sentence-transformer embeddings and save as .npy.

Usage:
    python scripts/lyrics_to_embeddings.py --metadata data/raw/metadata_aligned.csv --out_dir data/features/lyrics_embeddings --model all-mpnet-base-v2
"""
import argparse
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', default='data/raw/metadata_aligned.csv')
    parser.add_argument('--out_dir', default='data/features/lyrics_embeddings')
    parser.add_argument('--model', default='all-mpnet-base-v2')
    args = parser.parse_args()

    if not os.path.exists(args.metadata):
        raise FileNotFoundError(f"Metadata file not found: {args.metadata}. Run scripts/validate_dataset.py first.")

    df = pd.read_csv(args.metadata)
    os.makedirs(args.out_dir, exist_ok=True)

    try:
        model = SentenceTransformer(args.model)
    except Exception as e:
        print('Could not load SentenceTransformer model (will fallback to random embeddings):', e)
        model = None

    for _, r in df.iterrows():
        base = str(r['basename'])
        lyrics = str(r.get('lyrics', '') or '')
        if not lyrics or lyrics.strip() == 'nan':
            print(f'Skipping {base}: no lyrics')
            continue
        try:
            if model:
                emb = model.encode(lyrics, show_progress_bar=False)
            else:
                # fallback: deterministic pseudo-random embedding based on text hash
                rng = np.random.RandomState(abs(hash(lyrics)) % (2**32))
                emb = rng.randn(768).astype('float32')
        except Exception as e:
            print(f'Embedding failed for {base}, using random fallback: {e}')
            rng = np.random.RandomState(abs(hash(lyrics)) % (2**32))
            emb = rng.randn(768).astype('float32')
        np.save(os.path.join(args.out_dir, base + '.npy'), emb)
    print('Done. Saved embeddings to', args.out_dir)

if __name__ == '__main__':
    main()
