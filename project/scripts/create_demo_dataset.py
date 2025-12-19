"""
Create a small synthetic demo dataset with audio and lyrics to run the full pipeline without requiring external downloads.
Generates sine-wave audio files and simple lyrics files, plus a metadata CSV.

Usage:
    python scripts/create_demo_dataset.py --n 12 --out_dir data/raw
"""
import os
import argparse
import numpy as np
import soundfile as sf
import csv


def make_sine(freq, duration=3.0, sr=22050):
    t = np.linspace(0, duration, int(sr*duration), endpoint=False)
    y = 0.3 * np.sin(2 * np.pi * freq * t)
    return y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=12)
    parser.add_argument('--out_dir', default='data/raw')
    parser.add_argument('--sr', type=int, default=22050)
    args = parser.parse_args()

    audio_dir = os.path.join(args.out_dir, 'audio')
    lyrics_dir = os.path.join(args.out_dir, 'lyrics')
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(lyrics_dir, exist_ok=True)

    rows = []

    freqs = np.linspace(220, 880, args.n)
    languages = ['english', 'bangla']

    for i, f in enumerate(freqs):
        base = f'demo_{i:03d}'
        fname = base + '.wav'
        y = make_sine(f, duration=3.0, sr=args.sr)
        sf.write(os.path.join(audio_dir, fname), y, args.sr)

        # simple synthetic lyrics (vary language occasionally)
        lang = languages[i % len(languages)]
        lyrics = f"This is demo track {i}. Language: {lang}. Frequency {int(f)} Hz.\n" * 3
        with open(os.path.join(lyrics_dir, base + '.txt'), 'w', encoding='utf-8') as fh:
            fh.write(lyrics)

        rows.append({'basename': base, 'filename': os.path.join('audio', fname), 'lyrics': lyrics, 'label': lang})

    # write metadata CSV
    meta_path = os.path.join(args.out_dir, 'metadata.csv')
    with open(meta_path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=['basename', 'filename', 'lyrics', 'label'])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f'Created {args.n} demo tracks in {args.out_dir}')

if __name__ == '__main__':
    main()
