# Latent Visualization

This directory contains 2D projections of learned latent representations from VAE models.

## Files

### assignments.csv
Per-sample cluster assignments with metadata:
- **file**: Original feature filename
- **sample_name**: Track identifier
- **true_genre**: Ground-truth music genre label
- **predicted_cluster**: K-Means cluster assignment (K=2)
- **latent_distance**: Euclidean distance to cluster centroid
- **confidence**: Softmax probability for assigned cluster (higher = more confident)

### pca_scatter.png
PCA 2D projection of latent space (16-dimensional → 2D):
- Principal Component Analysis preserves maximum variance
- Colors represent K-Means cluster assignments
- Shows global structure and cluster separation

### tsne_scatter.png
t-SNE 2D projection of latent space (16-dimensional → 2D):
- t-Distributed Stochastic Neighbor Embedding preserves local structure
- Colors represent K-Means cluster assignments
- Perplexity=10, learning_rate=200, iterations=1000
- Better visualizes non-linear relationships than PCA

## Clustering Results

Best configuration (K=2):
- **Silhouette Score**: 0.845 (excellent cluster quality)
- **Davies-Bouldin Index**: 0.521 (low is better)
- **ARI**: 1.0 (perfect agreement with ground-truth)
- **NMI**: 1.0 (perfect information sharing)

Cluster distribution:
- Cluster 0: 10 samples (pop genre)
- Cluster 1: 2 samples (rock genre)

## Regenerate

To reproduce these visualizations:

```bash
python -m project.src.generate_results
```

This will:
1. Load features from `data/features/*.npy`
2. Train K-Means on VAE latent embeddings
3. Generate PCA and t-SNE projections
4. Save cluster assignments and plots
