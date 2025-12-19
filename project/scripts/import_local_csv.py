"""
Import a local CSV into the project's data/raw structure.

Usage:
  python project/scripts/import_local_csv.py --infile "D:/Photo self/artists-data.csv" --out project/data/raw/metadata.csv --write_texts

This will create:
 - project/data/raw/metadata.csv
 - project/data/raw/metadata_aligned.csv
 - project/data/raw/lyrics/<basename>.txt (if --write_texts)

The script tries to auto-detect a lyrics column and an id/title column.
"""
import argparse
import os
import csv
import pandas as pd
import re


def pick_lyrics_column(df):
    lower_cols = [c.lower() for c in df.columns]
    candidates = ['lyrics', 'text', 'song', 'song_lyrics', 'lyrics_text', 'body']
    for c in candidates:
        if c in lower_cols:
            return df.columns[lower_cols.index(c)]
    # fallback: look for a long text-like column
    for c in df.columns:
        if df[c].dtype == object and df[c].astype(str).map(len).median() > 20:
            return c
    return None


def pick_id_column(df):
    lower_cols = [c.lower() for c in df.columns]
    for c in ['id', 'track_id', 'song_id', 'title', 'name']:
        if c in lower_cols:
            return df.columns[lower_cols.index(c)]
    return None


def sanitize_basename(s):
    s = str(s)
    s = s.strip()
    s = re.sub(r'\s+', '_', s)
    s = re.sub(r'[^A-Za-z0-9_\-]', '', s)
    if not s:
        s = 'song'
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', required=True, help='Path to local CSV file')
    parser.add_argument('--out', default='project/data/raw/metadata.csv', help='Output metadata CSV path')
    parser.add_argument('--write_texts', action='store_true', help='Also write per-track .txt files to data/raw/lyrics')
    args = parser.parse_args()

    if not os.path.exists(args.infile):
        print('Input file not found:', args.infile)
        return

    df = pd.read_csv(args.infile)
    if df.empty:
        print('Input CSV is empty:', args.infile)
        return

    lyrics_col = pick_lyrics_column(df)
    if lyrics_col is None:
        print('Could not detect a lyrics/text column. Columns available:', list(df.columns))
        return
    id_col = pick_id_column(df)
    print('Detected lyrics column:', lyrics_col, 'id/title column:', id_col)

    out_dir = os.path.dirname(args.out)
    os.makedirs(out_dir, exist_ok=True)
    lyrics_dir = os.path.join(out_dir, 'lyrics')
    if args.write_texts:
        os.makedirs(lyrics_dir, exist_ok=True)

    rows = []
    for i, r in df.reset_index(drop=True).iterrows():
        lyrics = str(r.get(lyrics_col, '') or '')
        if id_col:
            base = sanitize_basename(r.get(id_col, '') or f'kaggle_{i:06d}')
        else:
            base = f'kaggle_{i:06d}'
        filename = ''
        rows.append({'basename': base, 'filename': filename, 'lyrics': lyrics, 'language': str(r.get('language', '') or '')})
        if args.write_texts:
            try:
                with open(os.path.join(lyrics_dir, base + '.txt'), 'w', encoding='utf-8') as fh:
                    fh.write(lyrics)
            except Exception as e:
                print('Could not write lyrics for', base, e)

    fieldnames = ['basename', 'filename', 'lyrics', 'language']
    with open(args.out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    aligned_path = os.path.join(out_dir, 'metadata_aligned.csv')
    with open(aligned_path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print('Wrote', args.out, 'and', aligned_path, 'and', len(rows), 'entries')


if __name__ == '__main__':
    main()
