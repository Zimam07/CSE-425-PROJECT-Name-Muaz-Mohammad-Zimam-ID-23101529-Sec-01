"""
Cluster VAE latents and produce evaluation metrics and visualizations (t-SNE, UMAP, PCA).

Usage:
    python scripts/cluster_and_visualize.py --latents results/demo_vae/latents.npy --k 2 --out_dir results/demo_analysis

Outputs:
 - results/demo_analysis/metrics.csv
 - results/demo_analysis/tsne.png
 - results/demo_analysis/umap.png
 - results/demo_analysis/pca.png
 - results/demo_analysis/assignments.csv
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# ensure src imports work
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.eval_metrics import compute_metrics


def save_scatter(X2, labels, path, title):
    plt.figure(figsize=(6,6))
    palette = sns.color_palette('tab10', n_colors=len(np.unique(labels)))
    sns.scatterplot(x=X2[:,0], y=X2[:,1], hue=labels, palette=palette, legend='brief', s=60)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latents', required=True)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--clusterer', choices=['kmeans','agglomerative','dbscan'], default='kmeans')
    parser.add_argument('--out_dir', default='results/demo_analysis')
    parser.add_argument('--metadata', type=str, default=None, help='Optional metadata CSV with basename,label columns to compute supervised metrics')
    parser.add_argument('--feat_dir', type=str, default=None, help='Optional feature dir to recover basenames for ordering')
    args = parser.parse_args()

    X = np.load(args.latents)
    os.makedirs(args.out_dir, exist_ok=True)

    # PCA for quick visualization
    pca = PCA(n_components=min(2, X.shape[1]))
    Xpca = pca.fit_transform(X)

    # Choose clustering algorithm for PCA-reduced space
    from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
    clusterer = args.clusterer.lower()
    if clusterer == 'kmeans':
        model = KMeans(n_clusters=args.k, random_state=123)
        pred = model.fit_predict(Xpca)
    elif clusterer == 'agglomerative':
        model = AgglomerativeClustering(n_clusters=args.k)
        pred = model.fit_predict(Xpca)
    elif clusterer == 'dbscan':
        model = DBSCAN()
        pred = model.fit_predict(Xpca)
    else:
        raise ValueError('Unknown clusterer: ' + args.clusterer)

    # recover true labels if metadata provided
    labels_true = None
    if args.metadata and args.feat_dir:
        import pandas as pd
        df = pd.read_csv(args.metadata)
        # build mapping basename -> label (if label column present)
        if 'basename' in df.columns and 'label' in df.columns:
            label_map = {str(r['basename']): r['label'] for _, r in df.iterrows()}
            # if feat_dir provided, sort files to match training ordering
            files = sorted([f for f in os.listdir(args.feat_dir) if f.endswith('.npy')])
            labels_true = []
            for f in files:
                base = os.path.splitext(f)[0]
                labels_true.append(label_map.get(base, ''))
            # if labels are strings, encode them numerically
            if labels_true and isinstance(labels_true[0], str):
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                labels_true = le.fit_transform(labels_true)
        else:
            print('Metadata does not contain basename/label columns; skipping supervised metrics')

    # Compute metrics (use PCA features for cluster metrics)
    metrics = compute_metrics(Xpca, pred, labels_true=labels_true)
    metrics['clusterer'] = args.clusterer

    # Save metrics
    import csv
    with open(os.path.join(args.out_dir, 'metrics.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['metric', 'value'])
        for k, v in metrics.items():
            w.writerow([k, v])

    # Save assignments
    import pandas as pd
    df = pd.DataFrame({'idx': np.arange(len(pred)), 'cluster': pred})
    df.to_csv(os.path.join(args.out_dir, 'assignments.csv'), index=False)

    # If metadata and feat_dir provided, compute cluster distribution over languages (if available)
    if args.metadata and args.feat_dir:
        try:
            dfmeta = pd.read_csv(args.metadata)
            # build map basename -> language if column exists
            lang_map = {}
            if 'basename' in dfmeta.columns and 'language' in dfmeta.columns:
                lang_map = dict(zip(dfmeta['basename'].astype(str), dfmeta['language'].astype(str)))
            else:
                # fallback: attempt to parse language from lyrics column if present
                if 'basename' in dfmeta.columns and 'lyrics' in dfmeta.columns:
                    import re
                    for _, r in dfmeta.iterrows():
                        base = str(r['basename'])
                        lyrics = str(r['lyrics'])
                        m = re.search(r'Language:\s*([A-Za-z\-]+)', lyrics, re.IGNORECASE)
                        if m:
                            lang_map[base] = m.group(1).lower()
            # recover basenames in same order as features in feat_dir
            files = sorted([f for f in os.listdir(args.feat_dir) if f.endswith('.npy')])
            langs = [lang_map.get(os.path.splitext(f)[0], '') for f in files]
            # only proceed if we have at least one non-empty language
            if any(l for l in langs):
                # build contingency table
                import collections
                counts = {}
                for i, cl in enumerate(pred):
                    if i < len(langs):
                        lang = langs[i] or 'unknown'
                        counts.setdefault(cl, {}).setdefault(lang, 0)
                        counts[cl][lang] += 1
                # build dataframe for plotting
                import matplotlib
                langs_all = sorted(set([la for sub in counts.values() for la in sub.keys()]))
                rows = []
                for cl, sub in sorted(counts.items()):
                    row = {'cluster': cl}
                    for la in langs_all:
                        row[la] = sub.get(la, 0)
                    rows.append(row)
                cdf = pd.DataFrame(rows)
                cdf = cdf.set_index('cluster')
                # stacked bar plot
                ax = cdf.plot(kind='bar', stacked=True, figsize=(8,4), colormap='tab20')
                ax.set_xlabel('Cluster')
                ax.set_ylabel('Count')
                ax.set_title('Cluster distribution by language')
                plt.tight_layout()
                plt.savefig(os.path.join(args.out_dir, 'cluster_language_dist.png'))
                plt.close()
        except Exception:
            print('Failed to compute cluster-language distribution; continuing...')

    # t-SNE (choose perplexity safely for small datasets)
    n_samples = Xpca.shape[0]
    perp = min(30, max(2, (n_samples - 1) // 3))
    tsne = TSNE(n_components=2, random_state=123, perplexity=perp)
    Xtsne = tsne.fit_transform(Xpca)
    save_scatter(Xtsne, pred, os.path.join(args.out_dir, 'tsne.png'), 't-SNE of VAE latents (PCA->t-SNE)')

    # UMAP (if available)
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=123)
        Xumap = reducer.fit_transform(Xpca)
        save_scatter(Xumap, pred, os.path.join(args.out_dir, 'umap.png'), 'UMAP of VAE latents')
    except Exception:
        print('UMAP not available; skipped UMAP plot')

    # PCA scatter
    save_scatter(Xpca, pred, os.path.join(args.out_dir, 'pca.png'), 'PCA of VAE latents')

    print('Saved clustering metrics and visualizations to', args.out_dir)


if __name__ == '__main__':
    main()
