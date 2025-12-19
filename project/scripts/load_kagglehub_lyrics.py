"""
Load a lyrics dataset from Kaggle using kagglehub and produce
`data/raw/metadata.csv` and `data/raw/lyrics/*.txt` (optional) so the
existing pipeline can consume it.

Usage examples:
  python scripts/load_kagglehub_lyrics.py --dataset neisse/scrapped-lyrics-from-6-genres
  python scripts/load_kagglehub_lyrics.py --dataset owner/dataset --file_path lyrics.csv

Notes:
 - Requires `kagglehub[pandas-datasets]` (or the Kaggle CLI as a fallback).
 - If kagglehub fails, this will try to fall back to the `kaggle` CLI and
   read any CSV files found in the downloaded dataset folder.
"""
import argparse
import os
import sys
import csv
import pandas as pd


def try_load_with_kagglehub(dataset_slug, file_path):
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter
        print('Using kagglehub to load dataset', dataset_slug, 'file:', file_path)
        df = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, dataset_slug, file_path)
        if df is None:
            raise RuntimeError('kagglehub returned None')
        return df
    except Exception as e:
        print('kagglehub load failed:', e)
        return None


def try_download_with_kaggle_cli(dataset_slug, out_dir):
    # fallback to kaggle CLI: `kaggle datasets download -d <slug> -p <out_dir> --unzip`
    import subprocess
    os.makedirs(out_dir, exist_ok=True)
    cmd = ['kaggle', 'datasets', 'download', '-d', dataset_slug, '-p', out_dir, '--unzip']
    print('Falling back to kaggle CLI: running', ' '.join(cmd))
    subprocess.run(cmd, check=True)
    # find csv files in out_dir
    csvs = []
    for root, _, files in os.walk(out_dir):
        for f in files:
            if f.lower().endswith('.csv'):
                csvs.append(os.path.join(root, f))
    return csvs


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


def save_metadata_and_texts(df, lyrics_col, id_col, out_dir, write_texts=True):
    os.makedirs(out_dir, exist_ok=True)
    lyrics_dir = os.path.join(os.path.dirname(out_dir), 'lyrics')
    if write_texts:
        os.makedirs(lyrics_dir, exist_ok=True)

    rows = []
    for i, r in df.reset_index(drop=True).iterrows():
        lyrics = str(r.get(lyrics_col, '') or '')
        if id_col:
            base = str(r.get(id_col, '') or '')
            if base:
                base = base.replace(' ', '_')
        else:
            base = f'kaggle_{i:06d}'
        # ensure a non-empty basename
        if not base:
            base = f'kaggle_{i:06d}'
        filename = ''  # no audio by default
        rows.append({'basename': base, 'filename': filename, 'lyrics': lyrics, 'language': ''})
        if write_texts:
            try:
                with open(os.path.join(lyrics_dir, base + '.txt'), 'w', encoding='utf-8') as fh:
                    fh.write(lyrics)
            except Exception as e:
                print('Could not write lyrics text for', base, e)
    # write metadata.csv and metadata_aligned.csv so downstream scripts work
    meta_path = os.path.join(out_dir)
    fieldnames = ['basename', 'filename', 'lyrics', 'language']
    with open(meta_path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    # It's convenient to also write metadata_aligned.csv (same content since no audio)
    aligned_path = os.path.join(os.path.dirname(out_dir), 'metadata_aligned.csv')
    with open(aligned_path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print('Wrote', meta_path, 'and', aligned_path, 'and', len(rows), 'entries')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Kaggle dataset slug (owner/dataset-name)')
    parser.add_argument('--file', default='', help='Optional file inside the dataset to load (CSV).')
    parser.add_argument('--out', default='data/raw/metadata.csv', help='Output metadata CSV (metadata.csv)')
    parser.add_argument('--write_texts', action='store_true', help='Also write per-track .txt files to data/raw/lyrics')
    args = parser.parse_args()

    df = None
    if args.file:
        df = try_load_with_kagglehub(args.dataset, args.file)
    else:
        # try to load without file path (load latest or main CSV)
        df = try_load_with_kagglehub(args.dataset, '')

    if df is None:
        # fallback: download dataset and try to read CSVs
        tmp_dir = os.path.join('data', 'raw', args.dataset.replace('/', '_'))
        try:
            csvs = try_download_with_kaggle_cli(args.dataset, tmp_dir)
            if not csvs:
                print('No CSV files found in downloaded dataset', args.dataset)
                sys.exit(1)
            # pick the largest CSV (heuristic)
            csvs.sort(key=lambda p: os.path.getsize(p), reverse=True)
            path = csvs[0]
            print('Reading CSV from', path)
            df = pd.read_csv(path)
        except Exception as e:
            print('Failed to download/read dataset:', e)
            sys.exit(1)

    # now we have a DataFrame
    lyrics_col = pick_lyrics_column(df)
    if lyrics_col is None:
        print('Could not detect a lyrics/text column in the dataset. Columns available:', list(df.columns))
        sys.exit(1)
    id_col = pick_id_column(df)
    print('Detected lyrics column:', lyrics_col, 'id column:', id_col)

    save_metadata_and_texts(df, lyrics_col, id_col, args.out, write_texts=args.write_texts)


if __name__ == '__main__':
    main()
