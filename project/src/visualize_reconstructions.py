"""
Visualization of VAE reconstructions: original vs. reconstructed features.

Usage:
    python src/visualize_reconstructions.py --model_path results/demo_vae/model_epoch10.pt \
                                             --features_dir data/features \
                                             --n_samples 12 \
                                             --output_dir results/reconstructions
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

# ensure src imports work
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.vae import VAE, ConvVAE, BetaVAE, ConditionalVAE


def load_features(features_dir, n_samples=None):
    """Load .npy feature files from directory."""
    files = sorted([f for f in os.listdir(features_dir) if f.endswith('.npy')])
    if n_samples:
        files = files[:n_samples]
    
    features = []
    for f in files:
        data = np.load(os.path.join(features_dir, f))
        features.append(data)
    
    return np.array(features), files


def plot_reconstructions(model, features, model_name='VAE', output_path='results/reconstructions.png'):
    """
    Create detailed reconstruction comparison plots.
    
    Arguments:
        model: trained VAE model
        features: (n_samples, feature_dim) array
        model_name: name of model for title
        output_path: where to save the figure
    """
    model.eval()
    n_samples = min(12, len(features))
    
    # Prepare data
    X = torch.FloatTensor(features[:n_samples])
    
    # Get reconstructions
    with torch.no_grad():
        if isinstance(model, ConditionalVAE):
            # For CVAE, use dummy labels (0)
            y = torch.zeros(n_samples, dtype=torch.long)
            recon, mu, logvar = model(X, y)
        else:
            recon, mu, logvar = model(X)
        
        recon = recon.cpu().numpy()
        mu = mu.cpu().numpy()
    
    # Calculate reconstruction errors
    recon_errors = np.mean((features[:n_samples] - recon) ** 2, axis=1)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(n_samples, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    for i in range(n_samples):
        # Original
        ax_orig = fig.add_subplot(gs[i, 0])
        ax_orig.plot(features[i], 'b-', linewidth=2, label='Original')
        ax_orig.set_ylabel(f'Sample {i}')
        if i == 0:
            ax_orig.set_title('Original Features', fontsize=12, fontweight='bold')
        ax_orig.grid(True, alpha=0.3)
        ax_orig.set_ylim([features[i].min() - 0.5, features[i].max() + 0.5])
        
        # Reconstructed
        ax_recon = fig.add_subplot(gs[i, 1])
        ax_recon.plot(recon[i], 'r-', linewidth=2, label='Reconstructed')
        if i == 0:
            ax_recon.set_title('Reconstructed Features', fontsize=12, fontweight='bold')
        ax_recon.grid(True, alpha=0.3)
        ax_recon.set_ylim([features[i].min() - 0.5, features[i].max() + 0.5])
        
        # Difference
        ax_diff = fig.add_subplot(gs[i, 2])
        diff = np.abs(features[i] - recon[i])
        ax_diff.bar(range(len(diff)), diff, color='orange', alpha=0.7)
        ax_diff.set_title(f'Absolute Error (MSE: {recon_errors[i]:.4f})', fontsize=10) if i == 0 else None
        ax_diff.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'{model_name} - Reconstruction Comparison\n(Original vs Reconstructed vs Error)', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved reconstruction comparison to {output_path}")
    plt.close()
    
    return recon_errors


def plot_reconstruction_metrics(model, features, model_name='VAE', output_path='results/reconstruction_metrics.png'):
    """
    Create metrics plots for reconstruction quality.
    
    Arguments:
        model: trained VAE model
        features: (n_samples, feature_dim) array
        model_name: name of model
        output_path: where to save figure
    """
    model.eval()
    
    # Get reconstructions
    X = torch.FloatTensor(features)
    with torch.no_grad():
        if isinstance(model, ConditionalVAE):
            y = torch.zeros(len(features), dtype=torch.long)
            recon, mu, logvar = model(X, y)
        else:
            recon, mu, logvar = model(X)
        recon = recon.cpu().numpy()
    
    # Calculate metrics per sample
    per_sample_mse = np.mean((features - recon) ** 2, axis=1)
    per_sample_mae = np.mean(np.abs(features - recon), axis=1)
    
    # Per-feature reconstruction error
    per_feature_mse = np.mean((features - recon) ** 2, axis=0)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # MSE per sample
    axes[0, 0].hist(per_sample_mse, bins=20, color='blue', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(np.mean(per_sample_mse), color='red', linestyle='--', linewidth=2, label='Mean')
    axes[0, 0].set_xlabel('MSE per Sample')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'Reconstruction MSE Distribution\nMean: {np.mean(per_sample_mse):.4f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE per sample
    axes[0, 1].hist(per_sample_mae, bins=20, color='green', alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(np.mean(per_sample_mae), color='red', linestyle='--', linewidth=2, label='Mean')
    axes[0, 1].set_xlabel('MAE per Sample')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'Reconstruction MAE Distribution\nMean: {np.mean(per_sample_mae):.4f}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # MSE per feature
    axes[1, 0].bar(range(len(per_feature_mse)), per_feature_mse, color='purple', alpha=0.7)
    axes[1, 0].set_xlabel('Feature Index')
    axes[1, 0].set_ylabel('MSE')
    axes[1, 0].set_title('Reconstruction MSE by Feature')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Cumulative statistics
    sorted_mse = np.sort(per_sample_mse)
    cumsum = np.cumsum(sorted_mse) / np.sum(sorted_mse)
    axes[1, 1].plot(cumsum, linewidth=2, color='darkblue')
    axes[1, 1].fill_between(range(len(cumsum)), cumsum, alpha=0.3)
    axes[1, 1].set_xlabel('Sample Index (sorted by MSE)')
    axes[1, 1].set_ylabel('Cumulative MSE Fraction')
    axes[1, 1].set_title('Cumulative Reconstruction Error')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'{model_name} - Reconstruction Quality Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved reconstruction metrics to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize VAE reconstructions')
    parser.add_argument('--model_path', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--features_dir', required=True, help='Directory containing feature .npy files')
    parser.add_argument('--model_type', choices=['vae', 'conv_vae', 'beta_vae', 'cvae'], 
                       default='vae', help='Type of VAE model')
    parser.add_argument('--latent_dim', type=int, default=32, help='Latent dimension')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension (for non-conv models)')
    parser.add_argument('--n_samples', type=int, default=12, help='Number of samples to visualize')
    parser.add_argument('--output_dir', default='results/reconstructions', help='Output directory')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load features
    print(f"Loading features from {args.features_dir}...")
    features, files = load_features(args.features_dir, args.n_samples)
    print(f"✓ Loaded {len(features)} features")
    
    # Determine input dimension
    input_dim = features.shape[1]
    print(f"Feature dimension: {input_dim}")
    
    # Load model
    print(f"Loading {args.model_type.upper()} model from {args.model_path}...")
    if args.model_type == 'vae':
        model = VAE(input_dim=input_dim, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim)
    elif args.model_type == 'conv_vae':
        model = ConvVAE(input_dim=input_dim, latent_dim=args.latent_dim)
    elif args.model_type == 'beta_vae':
        model = BetaVAE(input_dim=input_dim, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim)
    elif args.model_type == 'cvae':
        model = ConditionalVAE(input_dim=input_dim, n_classes=10, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim)
    
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    print(f"✓ Loaded model checkpoint")
    
    # Generate visualizations
    print("\nGenerating reconstruction visualizations...")
    recon_errors = plot_reconstructions(
        model, features, 
        model_name=f'{args.model_type.upper()} Model',
        output_path=os.path.join(args.output_dir, f'{args.model_type}_reconstructions.png')
    )
    
    print("Generating reconstruction metrics...")
    plot_reconstruction_metrics(
        model, features,
        model_name=f'{args.model_type.upper()} Model',
        output_path=os.path.join(args.output_dir, f'{args.model_type}_metrics.png')
    )
    
    # Print statistics
    print("\n" + "="*60)
    print(f"Reconstruction Error Statistics for {args.model_type.upper()}:")
    print("="*60)
    print(f"Mean MSE:     {np.mean(recon_errors):.6f}")
    print(f"Median MSE:   {np.median(recon_errors):.6f}")
    print(f"Std Dev:      {np.std(recon_errors):.6f}")
    print(f"Min MSE:      {np.min(recon_errors):.6f}")
    print(f"Max MSE:      {np.max(recon_errors):.6f}")
    print("="*60)
    
    print(f"\n✅ Visualizations saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
