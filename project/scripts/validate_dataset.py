"""
Validate dataset layout and build a canonical metadata CSV that links audio files and lyrics.

Usage:
    python scripts/validate_dataset.py --audio_dir data/raw/audio --lyrics_dir data/raw/lyrics --metadata data/raw/metadata.csv --out data/raw/metadata_aligned.csv

The script will:
 - Scan audio files and lyrics files
 - If a metadata CSV is provided, try to match filenames or track ids
 - Produce `metadata_aligned.csv` with columns: basename, filename, lyrics (if available)
"""
import argparse
import os
import csv
import pandas as pd


def scan_audio(audio_dir):
    files = []
    for root, _, fs in os.walk(audio_dir):
        for f in fs:
            if f.lower().endswith(('.mp3', '.wav', '.flac', '.ogg', '.m4a')):
                files.append(os.path.join(root, f))
    return files


def scan_lyrics(lyrics_dir):
    texts = {}
    if not os.path.exists(lyrics_dir):
        return texts
    for root, _, fs in os.walk(lyrics_dir):
        for f in fs:
            if f.lower().endswith('.txt'):
                base = os.path.splitext(f)[0]
                try:
                    with open(os.path.join(root, f), 'r', encoding='utf-8') as fh:
                        texts[base] = fh.read().strip()
                except Exception:
                    texts[base] = ''
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', default='data/raw/audio')
    parser.add_argument('--lyrics_dir', default='data/raw/lyrics')
    parser.add_argument('--metadata', default='data/raw/metadata.csv')
    parser.add_argument('--out', default='data/raw/metadata_aligned.csv')
    args = parser.parse_args()

    audio_files = scan_audio(args.audio_dir)
    audio_map = {os.path.splitext(os.path.basename(p))[0]: p for p in audio_files}
    lyrics_map = scan_lyrics(args.lyrics_dir)

    rows = []

    # detect whether metadata contains extra columns like 'label' or 'language'
    label_present = False
    language_present = False

    # helper to extract language from lyrics text if available
    import re
    def extract_language(lyrics_text):
        if not lyrics_text:
            return ''
        m = re.search(r'Language:\s*([A-Za-z\-]+)', lyrics_text, re.IGNORECASE)
        if m:
            return m.group(1).lower()
        return ''

    # If metadata CSV exists, try to use it for matching
    if os.path.exists(args.metadata):
        try:
            df = pd.read_csv(args.metadata)
            # look for common columns
            fname_col = None
            for c in ['filename', 'file', 'file_name', 'track', 'track_id']:
                if c in df.columns:
                    fname_col = c
                    break
            if 'label' in df.columns:
                label_present = True
            if 'language' in df.columns:
                language_present = True
            for _, r in df.iterrows():
                fname = str(r[fname_col]) if fname_col else ''
                base = os.path.splitext(os.path.basename(fname))[0] if fname else ''
                audio_path = audio_map.get(base, '')
                lyrics = ''
                if 'lyrics' in df.columns:
                    lyrics = str(r['lyrics'])
                elif base in lyrics_map:
                    lyrics = lyrics_map[base]
                row = {'basename': base, 'filename': audio_path, 'lyrics': lyrics}
                if label_present:
                    row['label'] = r.get('label', '')
                # language from metadata column if present, else try to extract from lyrics
                if language_present:
                    row['language'] = str(r.get('language', '')).lower()
                else:
                    row['language'] = extract_language(row['lyrics'])
                rows.append(row)
        except Exception:
            print('Could not read metadata CSV; falling back to audio+lyrics files.')

    # add any audio files not covered
    covered = set([r['basename'] for r in rows])
    for base, path in audio_map.items():
        if base in covered:
            continue
        lyrics = lyrics_map.get(base, '')
        row = {'basename': base, 'filename': path, 'lyrics': lyrics}
        if label_present:
            row['label'] = ''
        # attempt to extract language from lyrics if metadata had none
        row['language'] = extract_language(row['lyrics'])
        rows.append(row)

    # Save
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fieldnames = ['basename', 'filename', 'lyrics']
    if label_present:
        fieldnames.append('label')
    # include language column (may be empty)
    fieldnames.append('language')
    with open(args.out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f'Wrote aligned metadata to {args.out}. Total tracks: {len(rows)}')

if __name__ == '__main__':
    main()
