"""
Prepare multimodal features by aligning audio feature .npy files and lyrics embeddings, and saving concatenated feature vectors.

Usage:
    python src/prepare_multimodal.py --audio_dir data/features --lyrics_dir data/features/lyrics_embeddings --out_dir data/features/multimodal
"""
import os
import argparse
import numpy as np


def load_np_map(dir_path):
    m = {}
    if not os.path.exists(dir_path):
        return m
    for f in os.listdir(dir_path):
        if f.endswith('.npy'):
            base = os.path.splitext(f)[0]
            m[base] = np.load(os.path.join(dir_path, f))
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', default='data/features')
    parser.add_argument('--lyrics_dir', default='data/features/lyrics_embeddings')
    parser.add_argument('--out_dir', default='data/features/multimodal')
    args = parser.parse_args()

    audio_map = load_np_map(args.audio_dir)
    lyrics_map = load_np_map(args.lyrics_dir)

    os.makedirs(args.out_dir, exist_ok=True)

    common = sorted(set(audio_map.keys()) & set(lyrics_map.keys()))
    if len(common) == 0:
        print('No overlapping keys between audio and lyrics embeddings. Consider checking paths or running feature extraction.')

    for base in common:
        a = audio_map[base]
        l = lyrics_map[base]
        # simple concatenation after standardization
        a = (a - a.mean()) / (a.std() + 1e-9)
        l = (l - l.mean()) / (l.std() + 1e-9)
        combined = np.concatenate([a, l])
        np.save(os.path.join(args.out_dir, base + '.npy'), combined)

    print(f'Saved {len(common)} multimodal feature files to {args.out_dir}')

if __name__ == '__main__':
    main()
