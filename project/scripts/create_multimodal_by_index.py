"""
Create multimodal features by pairing sorted audio feature files and lyrics embeddings by index.

Usage:
  python scripts/create_multimodal_by_index.py --audio_dir data/features --lyrics_dir data/features/lyrics_embeddings --out_dir data/features/multimodal_indexed

This is useful when audio and lyrics originate from the same dataset but basenames do not match.
"""
import os
import numpy as np
import argparse


def load_np_map(dir_path):
    m = {}
    if not os.path.exists(dir_path):
        return m
    for f in sorted(os.listdir(dir_path)):
        if f.endswith('.npy'):
            m[os.path.splitext(f)[0]] = np.load(os.path.join(dir_path, f))
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', default='data/features')
    parser.add_argument('--lyrics_dir', default='data/features/lyrics_embeddings')
    parser.add_argument('--out_dir', default='data/features/multimodal_indexed')
    args = parser.parse_args()

    audio_map = load_np_map(args.audio_dir)
    lyrics_map = load_np_map(args.lyrics_dir)

    aud_keys = sorted(audio_map.keys())
    lyr_keys = sorted(lyrics_map.keys())

    n = min(len(aud_keys), len(lyr_keys))
    if n == 0:
        print('No audio or lyrics embeddings available to pair.')
        return

    os.makedirs(args.out_dir, exist_ok=True)
    for i in range(n):
        a = audio_map[aud_keys[i]]
        l = lyrics_map[lyr_keys[i]]
        # standardize each
        a = (a - a.mean()) / (a.std() + 1e-9)
        l = (l - l.mean()) / (l.std() + 1e-9)
        combined = np.concatenate([a, l])
        outname = lyr_keys[i] + '.npy'  # use lyrics basename for consistency
        np.save(os.path.join(args.out_dir, outname), combined)
    print(f'Created {n} paired multimodal feature files in {args.out_dir}')

if __name__ == '__main__':
    main()
