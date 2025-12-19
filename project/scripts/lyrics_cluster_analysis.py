"""
Analyze lyrics embeddings: PCA scatter, silhouette plot, cluster sizes, save metrics.

Usage:
  python scripts/lyrics_cluster_analysis.py --feat_dir project/data/features/lyrics_embeddings_run --k 2 --out_dir results/lyrics_analysis
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import csv


def load_features(dir_path):
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
    parser.add_argument('--feat_dir', required=True)
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--out_dir', default='results/lyrics_analysis')
    args = parser.parse_args()

    X, names = load_features(args.feat_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    # PCA to 2D for plotting
    pca2 = PCA(n_components=2)
    X2 = pca2.fit_transform(X)

    model = KMeans(n_clusters=args.k, random_state=123)
    labels = model.fit_predict(X)

    # PCA scatter
    plt.figure(figsize=(6,6))
    for lab in np.unique(labels):
        mask = labels == lab
        plt.scatter(X2[mask,0], X2[mask,1], label=f'cluster {lab}', s=80)
    plt.legend()
    plt.title('PCA (2D) scatter of lyrics embeddings')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'pca_scatter.png'))
    plt.close()

    # silhouette
    if len(X) > 1 and args.k > 1:
        sil_score = silhouette_score(X, labels)
        sil_vals = silhouette_samples(X, labels)
        plt.figure(figsize=(6,4))
        plt.bar(range(len(sil_vals)), np.sort(sil_vals))
        plt.title(f'Silhouette values (score={sil_score:.3f})')
        plt.xlabel('Samples (sorted)')
        plt.ylabel('Silhouette value')
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, 'silhouette.png'))
        plt.close()
    else:
        sil_score = None

    # cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    plt.figure(figsize=(6,4))
    plt.bar(unique.astype(str), counts)
    plt.title('Cluster sizes')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'cluster_sizes.png'))
    plt.close()

    # Save metrics
    metrics = {
        'n_samples': len(X),
        'n_features': X.shape[1],
        'n_clusters': int(args.k),
        'silhouette': float(sil_score) if sil_score is not None else ''
    }
    with open(os.path.join(args.out_dir, 'metrics.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['metric','value'])
        for k,v in metrics.items():
            w.writerow([k,v])

    # Save assignments
    with open(os.path.join(args.out_dir, 'assignments.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['basename','cluster'])
        for name, lab in zip(names, labels):
            w.writerow([name, int(lab)])

    print('Saved plots and metrics to', args.out_dir)

if __name__ == '__main__':
    main()
