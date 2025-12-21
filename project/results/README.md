# Results

This folder contains artifacts generated from clustering and latent-space analysis.

Contents:
- `clustering_metrics.csv`: Aggregated clustering quality metrics across K values.
- `latent_visualization/`: Scatter plots and assignments produced from 2D projections.
  - `pca_scatter.png`: PCA 2D projection colored by K-Means clusters.
  - `tsne_scatter.png`: t-SNE 2D projection colored by K-Means clusters.
  - `assignments.csv`: Per-sample cluster assignments.

Regenerate everything:

```bash
# From repository root
python -m project.src.generate_results
```

Notes:
- Metrics include Silhouette, Davies-Bouldin, and Calinski-Harabasz.
- Best K is selected by the highest Silhouette score.
- Visualizations use PCA and t-SNE for intuitive 2D projections.
