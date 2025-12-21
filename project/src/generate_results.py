import os
from pathlib import Path
import csv
import glob
import numpy as np
from datetime import datetime

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_feature_matrix(features_dir: Path) -> tuple[list[str], np.ndarray]:
    npy_files = sorted(features_dir.glob("*.npy"))
    names = [f.name for f in npy_files]
    if not npy_files:
        raise FileNotFoundError(f"No .npy feature files found in {features_dir}")
    X_list = []
    for f in npy_files:
        arr = np.load(f)
        arr = arr.reshape(-1) if arr.ndim > 1 else arr
        X_list.append(arr)
    X = np.vstack(X_list)
    return names, X


def compute_metrics_for_k(X: np.ndarray, k_values: list[int]) -> list[dict]:
    results = []
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(Xs)
        sil = silhouette_score(Xs, labels)
        db = davies_bouldin_score(Xs, labels)
        ch = calinski_harabasz_score(Xs, labels)
        results.append({
            "K": k,
            "n_samples": Xs.shape[0],
            "silhouette": round(float(sil), 6),
            "davies_bouldin": round(float(db), 6),
            "calinski_harabasz": round(float(ch), 6),
        })
    return results


def save_metrics_csv(csv_path: Path, rows: list[dict]):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["timestamp", "K", "n_samples", "silhouette", "davies_bouldin", "calinski_harabasz"]
    now = datetime.now().isoformat(timespec="seconds")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            r_out = {"timestamp": now, **r}
            writer.writerow(r_out)


def best_k_by_silhouette(rows: list[dict]) -> int:
    return max(rows, key=lambda r: r["silhouette"])['K']


def visualize_latent(X: np.ndarray, names: list[str], k: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(Xs)

    # PCA 2D
    pca = PCA(n_components=2, random_state=42)
    Xp = pca.fit_transform(Xs)
    plt.figure(figsize=(6, 4))
    for lab in np.unique(labels):
        idx = labels == lab
        plt.scatter(Xp[idx, 0], Xp[idx, 1], s=60, alpha=0.8, label=f"Cluster {lab}")
    plt.title("PCA 2D Projection (colored by K-Means)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(loc="best", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "pca_scatter.png", dpi=150)
    plt.close()

    # t-SNE 2D (robust for small N)
    try:
        tsne = TSNE(n_components=2, perplexity=max(2, min(10, Xs.shape[0]-1)), random_state=42)
        Xt = tsne.fit_transform(Xs)
        plt.figure(figsize=(6, 4))
        for lab in np.unique(labels):
            idx = labels == lab
            plt.scatter(Xt[idx, 0], Xt[idx, 1], s=60, alpha=0.8, label=f"Cluster {lab}")
        plt.title("t-SNE 2D Projection (colored by K-Means)")
        plt.xlabel("t1")
        plt.ylabel("t2")
        plt.legend(loc="best", fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "tsne_scatter.png", dpi=150)
        plt.close()
    except Exception as e:
        # Optional: skip if t-SNE fails
        with (out_dir / "tsne_error.txt").open("w") as f:
            f.write(str(e))

    # Save assignments
    with (out_dir / "assignments.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "cluster"])
        for name, lab in zip(names, labels):
            writer.writerow([name, int(lab)])


def main():
    repo_root = Path(__file__).resolve().parents[2]  # .../project/src/generate_results.py -> repo root
    project_dir = repo_root / "project"
    features_dir = project_dir / "data" / "features"
    results_dir = project_dir / "results"
    metrics_csv = results_dir / "clustering_metrics.csv"
    latvis_dir = results_dir / "latent_visualization"

    print("Loading features from:", features_dir)
    names, X = load_feature_matrix(features_dir)
    print(f"Loaded {len(names)} samples; feature dim = {X.shape[1]}")

    print("Computing clustering metrics for K in [2, 3, 4, 5]")
    rows = compute_metrics_for_k(X, k_values=[2, 3, 4, 5])
    save_metrics_csv(metrics_csv, rows)
    best_k = best_k_by_silhouette(rows)
    print(f"Best K by silhouette: {best_k}")

    print("Generating latent visualizations (PCA, t-SNE)")
    visualize_latent(X, names, k=best_k, out_dir=latvis_dir)

    print("Done. Outputs:")
    print(" -", metrics_csv)
    print(" -", latvis_dir / "pca_scatter.png")
    print(" -", latvis_dir / "tsne_scatter.png")
    print(" -", latvis_dir / "assignments.csv")


if __name__ == "__main__":
    main()
