import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

# Ensure project root is on sys.path so imports work when running from scripts
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.dataset import FeatureDataset
from src.vae import VAE
from src.conv_vae import ConvVAE


def train(args):
    if args.model == 'mlp':
        dataset = FeatureDataset(args.feat_dir)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        sample = dataset[0]
        input_dim = sample.shape[0]

        model = VAE(input_dim, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim)
    elif args.model == 'conv':
        from src.spectrogram_dataset import SpectrogramDataset
        dataset = SpectrogramDataset(os.path.join(args.feat_dir, 'spec'))
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        model = ConvVAE(in_channels=1, hidden_channels=args.hidden_channels, latent_dim=args.latent_dim, kernel_size=args.kernel_size)
    else:
        raise ValueError('Unknown model type; choose mlp or conv')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.out_dir, exist_ok=True)

    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss = model.loss_function(recon, batch, mu, logvar, beta=args.beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}/{args.epochs} - Loss: {total_loss/len(dataset):.4f}")
        torch.save(model.state_dict(), os.path.join(args.out_dir, f'model_epoch{epoch}.pt'))

    # Save final latent embeddings
    model.eval()
    latents = []
    with torch.no_grad():
        for i in range(len(dataset)):
            x = dataset[i].unsqueeze(0).to(device)
            enc_out = model.encode(x)
            # handle MLP VAE (mu, logvar) and ConvVAE (mu, logvar, shape)
            if isinstance(enc_out, tuple) and len(enc_out) == 2:
                mu, logvar = enc_out
            else:
                mu, logvar, _ = enc_out
            latents.append(mu.squeeze(0).cpu().numpy())
    import numpy as np
    np.save(os.path.join(args.out_dir, 'latents.npy'), np.stack(latents))
    print('Training finished. Latents saved to', args.out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat_dir', type=str, default='data/features')
    parser.add_argument('--out_dir', type=str, default='results')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--beta', type=float, default=1.0, help='Beta weight for KL term (beta-VAE)')
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'conv'], help='Model type: mlp (vector) or conv (spectrogram)')
    parser.add_argument('--hidden_channels', type=int, default=32, help='Hidden channels for ConvVAE')
    parser.add_argument('--kernel_size', type=int, default=4, help='Kernel size for ConvVAE convolutions')
    args = parser.parse_args()

    train(args)
