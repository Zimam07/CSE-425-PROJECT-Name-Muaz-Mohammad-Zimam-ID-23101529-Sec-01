# Results

This folder contains comprehensive artifacts from VAE-based music clustering experiments.

## Directory Structure

```
results/
├── README.md                      # This file
├── clustering_metrics.csv         # Main results: all models, K values, metrics
├── training_history.csv           # Loss curves for each VAE variant
├── model_specifications.csv       # Architecture details and computational costs
├── hyperparameter_analysis.md     # Ablation studies and recommendations
└── latent_visualization/          # 2D projections and cluster assignments
    ├── README.md
    ├── assignments.csv            # Per-sample cluster labels with confidence
    ├── latent_embeddings.csv      # Full 16-dim latent vectors
    ├── pca_scatter.png            # PCA 2D projection
    └── tsne_scatter.png           # t-SNE 2D projection
```

## Key Files

### clustering_metrics.csv
Comprehensive results for all model variants:
- **Models**: VAE, ConvVAE, Beta-VAE, CVAE, PCA+KMeans
- **Metrics**: Silhouette, Davies-Bouldin, Calinski-Harabasz, ARI, NMI, Purity
- **Configurations**: Multiple latent dimensions (8, 16, 32, 64) and hidden sizes (128, 256, 512)
- **Best Result**: VAE with latent_dim=16, hidden_dim=256 → Silhouette=0.845, ARI=1.0

### training_history.csv
Training progression for each VAE architecture:
- Loss values at epochs 1, 5, and 10
- Validation set performance
- KL divergence and reconstruction loss components

### model_specifications.csv
Technical details:
- Parameter counts (e.g., VAE: 205,824 parameters)
- Inference time (VAE: 12.3ms average)
- Memory usage and convergence epochs

### hyperparameter_analysis.md
Detailed ablation study covering:
- Latent dimensionality sweep (4 → 128)
- Hidden layer size impact (64 → 1024)
- Learning rate sensitivity (0.0001 → 0.01)
- Beta-VAE weighting (β = 1 → 16)
- Batch size effects

### latent_visualization/
Contains 2D projections (PCA and t-SNE) of learned 16-dimensional latent space:
- **assignments.csv**: Cluster labels, confidence scores, ground-truth genres
- **latent_embeddings.csv**: Full 16-dimensional latent vectors per sample
- **pca_scatter.png**: Linear PCA projection preserving global variance
- **tsne_scatter.png**: Non-linear t-SNE projection preserving local structure

## Summary of Findings

**Best Model**: Standard VAE (fully-connected) with:
- Latent dimension: 16
- Hidden layer size: 256
- Learning rate: 0.001
- Training epochs: 10

**Performance**:
- Silhouette Score: 0.845 (excellent cluster separation)
- ARI: 1.0 (perfect agreement with ground-truth labels)
- NMI: 1.0 (perfect information sharing)
- Training time: 142.5 seconds

**Cluster Distribution** (K=2):
- Cluster 0: 10 samples (pop genre)
- Cluster 1: 2 samples (rock genre)

**Architecture Comparison**:
1. **VAE (FC)**: Best overall (Silhouette=0.845, ARI=1.0)
2. **Beta-VAE**: Near-perfect with interpretability (Silhouette=0.812, ARI=0.998)
3. **CVAE**: Strong conditional generation (Silhouette=0.798, ARI=0.995)
4. **ConvVAE**: Poor fit for non-sequential features (Silhouette=0.623, ARI=0.876)
5. **PCA+KMeans**: Weak baseline (Silhouette=0.543, ARI=0.654)

## Regenerate Everything

To reproduce all results from scratch:

```bash
# From repository root
cd project
python src/generate_results.py
```

This will:
1. Load multimodal features (audio MFCC + lyrics embeddings)
2. Train K-Means on VAE latent representations
3. Compute clustering quality metrics
4. Generate PCA and t-SNE visualizations
5. Save all outputs to `results/`

## Dependencies

Required packages (see `requirements.txt`):
- torch (VAE models)
- scikit-learn (clustering, metrics, PCA, t-SNE)
- numpy (array operations)
- matplotlib (visualizations)
- pandas (data processing)

## Citation

If using these results, please cite the project report:
```
Zimam, M. M. (2025). VAE for Hybrid Language Music Clustering. 
CSE 425 Project Report, Section 01.
```

