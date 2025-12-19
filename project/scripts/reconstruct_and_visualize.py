"""
Reconstruct and visualize a few examples using a trained VAE or ConvVAE checkpoint.

Usage examples:
  python scripts/reconstruct_and_visualize.py --model_path results/demo_vae/model_epoch10.pt --model mlp --feat_dir data/features/multimodal --out_dir results/reconstructions --n 6
  python scripts/reconstruct_and_visualize.py --model_path results/conv_focus/conv_ld64_hc32/model_epoch30.pt --model conv --feat_dir data/features --out_dir results/reconstructions_conv --n 6

Produces per-sample images (original, reconstruction) and a small CSV with MSE per sample.
"""
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

# Ensure repo root for imports
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.vae import VAE
from src.conv_vae import ConvVAE


def load_numpy_files(dir_path, ext='.npy'):
    files = sorted([f for f in os.listdir(dir_path) if f.endswith(ext)])
    return files


def plot_and_save_vectors(orig, recon, path, title=None):
    plt.figure(figsize=(6,3))
    plt.plot(orig, label='orig')
    plt.plot(recon, label='recon', alpha=0.8)
    plt.legend()
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_and_save_spec(orig, recon, path, title=None):
    # orig/recon: numpy arrays shape (n_mels, n_frames)
    fig, axes = plt.subplots(1,2, figsize=(8,3))
    axes[0].imshow(orig, aspect='auto', origin='lower')
    axes[0].set_title('Original')
    axes[1].imshow(recon, aspect='auto', origin='lower')
    axes[1].set_title('Reconstruction')
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--model', choices=['mlp','conv'], required=True)
    parser.add_argument('--feat_dir', required=True, help='For mlp: data/features/multimodal; for conv: data/features (spec subdir)')
    parser.add_argument('--out_dir', default='results/reconstructions')
    parser.add_argument('--n', type=int, default=6)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--kernel_size', type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model == 'mlp':
        # load files
        files = load_numpy_files(args.feat_dir)
        if len(files) == 0:
            raise ValueError('No .npy features found in ' + args.feat_dir)
        # load sample to infer input_dim
        sample = np.load(os.path.join(args.feat_dir, files[0]))
        input_dim = sample.shape[0]
        model = VAE(input_dim, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device).eval()

        mses = []
        chosen = files[:args.n]
        for i, f in enumerate(chosen):
            x = np.load(os.path.join(args.feat_dir, f)).astype('float32')
            xt = torch.from_numpy(x).unsqueeze(0).to(device)
            with torch.no_grad():
                recon, mu, logvar = model(xt)
            rec = recon.squeeze(0).cpu().numpy()
            mse = float(((rec - x)**2).mean())
            mses.append((f, mse))
            plot_and_save_vectors(x, rec, os.path.join(args.out_dir, f'{os.path.splitext(f)[0]}_recon.png'), title=f)

    else:
        # conv: expect spectrogram .npy files under feat_dir/spec
        spec_dir = os.path.join(args.feat_dir, 'spec')
        files = load_numpy_files(spec_dir)
        if len(files) == 0:
            raise ValueError('No spectrogram .npy files found in ' + spec_dir)
        model = ConvVAE(in_channels=1, hidden_channels=args.hidden_channels, latent_dim=args.latent_dim, kernel_size=args.kernel_size)
        # Some ConvVAE checkpoints include layers that are lazily-initialized; attempt to load directly,
        # and if keys mismatch, run a dummy forward to initialize shapes and retry.
        state = torch.load(args.model_path, map_location=device)
        try:
            model.load_state_dict(state)
        except RuntimeError:
            # try to initialize by running a dummy forward using one sample
            sample_sp = np.load(os.path.join(spec_dir, files[0])).astype('float32')
            dummy = torch.from_numpy(sample_sp).unsqueeze(0).unsqueeze(0).to(device)
            model.to(device).eval()
            with torch.no_grad():
                try:
                    _ = model(dummy)
                except Exception:
                    pass
            model.load_state_dict(state)
        model.to(device).eval()

        mses = []
        chosen = files[:args.n]
        for i, f in enumerate(chosen):
            sp = np.load(os.path.join(spec_dir, f)).astype('float32')  # shape (n_mels, frames)
            xt = torch.from_numpy(sp).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                recon, mu, logvar = model(xt)
            rec = recon.squeeze(0).squeeze(0).cpu().numpy()
            # align shapes in case of small diff
            min_h = min(rec.shape[0], sp.shape[0])
            min_w = min(rec.shape[1], sp.shape[1])
            mse = float(((rec[:min_h,:min_w] - sp[:min_h,:min_w])**2).mean())
            mses.append((f, mse))
            plot_and_save_spec(sp, rec, os.path.join(args.out_dir, f'{os.path.splitext(f)[0]}_recon.png'), title=f)

    # Save mse summary
    import csv
    with open(os.path.join(args.out_dir, 'reconstruction_mse.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['file', 'mse'])
        for r in mses:
            w.writerow([r[0], r[1]])

    print('Saved reconstructions to', args.out_dir)


if __name__ == '__main__':
    main()
