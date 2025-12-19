"""
Quick clustering + evaluation script. Works on feature files in a directory or on saved latents (latents.npy).

Usage:
    python scripts/quick_eval.py --feat_dir data/features/multimodal --k 10 --out results/metrics.csv
"""
import argparse
import os
import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Ensure project root is on sys.path so we can import src.* when running scripts
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.eval_metrics import compute_metrics


def load_features_dir(dir_path):
    X = []
    names = []
    for f in sorted(os.listdir(dir_path)):
        if f.endswith('.npy'):
            names.append(os.path.splitext(f)[0])
            X.append(np.load(os.path.join(dir_path, f)))
    if len(X) == 0:
        raise ValueError('No .npy files found in ' + dir_path)
    return np.stack(X), names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat_dir', default='data/features/multimodal')
    parser.add_argument('--latents', default='')
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--clusterer', choices=['kmeans','agglomerative','dbscan'], default='kmeans')
    parser.add_argument('--out', default='results/metrics.csv')
    args = parser.parse_args()

    if args.latents and os.path.exists(args.latents):
        X = np.load(args.latents)
    else:
        X, names = load_features_dir(args.feat_dir)

    # baseline PCA -> clustering
    # Choose n_components safely for small sample sizes
    n_comp = min(50, X.shape[1], max(1, X.shape[0] - 1))
    pca = PCA(n_components=n_comp)
    Xp = pca.fit_transform(X)

    # Choose clustering algorithm
    from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
    clusterer = args.clusterer.lower()
    if clusterer == 'kmeans':
        model = KMeans(n_clusters=args.k, random_state=123)
        pred = model.fit_predict(Xp)
    elif clusterer == 'agglomerative':
        model = AgglomerativeClustering(n_clusters=args.k)
        pred = model.fit_predict(Xp)
    elif clusterer == 'dbscan':
        model = DBSCAN()
        pred = model.fit_predict(Xp)
    else:
        raise ValueError('Unknown clusterer: ' + args.clusterer)

    # Try to load labels if available (metadata_aligned.csv)
    labels_true = None
    try:
        import pandas as pd
        df = pd.read_csv('data/raw/metadata_aligned.csv')
        # if df has 'label' column map it
        if 'label' in df.columns:
            labels_true = [None]*len(pred)
            basename_to_label = dict(zip(df['basename'], df['label']))
            for i, name in enumerate(names):
                labels_true[i] = basename_to_label.get(name, None)
            if any(l is None for l in labels_true):
                labels_true = None
            else:
                labels_true = np.array(labels_true)
    except Exception:
        labels_true = None

    metrics = compute_metrics(Xp, pred, labels_true)
    metrics['clusterer'] = args.clusterer

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    import csv
    with open(args.out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['metric', 'value'])
        for k, v in metrics.items():
            w.writerow([k, v])

    print('Saved metrics to', args.out)

if __name__ == '__main__':
    main()
