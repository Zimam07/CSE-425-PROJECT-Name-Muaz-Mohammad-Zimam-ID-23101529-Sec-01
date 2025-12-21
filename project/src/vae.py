import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, latent_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc_decode(z))
        return self.fc_out(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    @staticmethod
    def loss_function(recon_x, x, mu, logvar, beta=1.0):
        """MSE reconstruction + beta * KL divergence (supports beta-VAE).

        Arguments:
            recon_x: reconstructed output
            x: target input
            mu, logvar: latent params
            beta: weight for the KL term (default 1.0)
        """
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kld_loss

class BetaVAE(VAE):
    """Beta-VAE for learning disentangled representations.
    
    Beta-VAE introduces a weight parameter (beta) on the KL divergence term
    to encourage learning of more interpretable latent factors.
    """
    
    def __init__(self, input_dim, hidden_dim=256, latent_dim=32, beta=4.0):
        super().__init__(input_dim, hidden_dim, latent_dim)
        self.beta = beta
    
    @staticmethod
    def loss_function(recon_x, x, mu, logvar, beta=4.0):
        """Beta-weighted VAE loss for disentangled representations.
        
        Arguments:
            recon_x: reconstructed output
            x: target input
            mu, logvar: latent params
            beta: weight for KL divergence (higher beta → more disentanglement)
        """
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kld_loss


class ConvVAE(nn.Module):
    """Convolutional VAE for 1D sequential audio features.
    
    Uses 1D convolutions for encoding/decoding audio feature sequences,
    more suitable for spectrograms and other sequential audio representations.
    """
    
    def __init__(self, input_dim, latent_dim=32, n_channels=16):
        """
        Arguments:
            input_dim: length of 1D feature sequence (e.g., 128)
            latent_dim: dimension of latent space
            n_channels: number of base channels in conv layers
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_channels = n_channels
        
        # Encoder: 1D convolutions with stride
        self.encoder = nn.Sequential(
            nn.Conv1d(1, n_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(n_channels, n_channels*2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(n_channels*2, n_channels*4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        
        # Calculate flattened dimension after convolutions (input_dim / 8)
        self.flat_dim = n_channels * 4 * (input_dim // 8)
        
        # Latent space
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)
        
        # Decoder setup
        self.fc_decode = nn.Linear(latent_dim, self.flat_dim)
        
        # Decoder: Transposed convolutions
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(n_channels*4, n_channels*2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(n_channels*2, n_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(n_channels, 1, kernel_size=4, stride=2, padding=1),
        )
    
    def encode(self, x):
        """Encode input to latent distribution parameters."""
        # Add channel dimension if not present
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_dim)
        
        # Convolutional encoding
        h = self.encoder(x)  # (batch, n_channels*4, input_dim/8)
        h = h.view(h.size(0), -1)  # Flatten
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Sample from latent distribution."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent vector to feature space."""
        h = self.fc_decode(z)
        h = h.view(h.size(0), self.n_channels*4, -1)  # Reshape for conv
        recon = self.decoder(h)
        return recon.squeeze(1)  # Remove channel dimension
    
    def forward(self, x):
        """Full VAE forward pass."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    @staticmethod
    def loss_function(recon_x, x, mu, logvar, beta=1.0):
        """ConvVAE loss function.
        
        Arguments:
            recon_x: reconstructed output
            x: target input
            mu, logvar: latent distribution parameters
            beta: weight for KL divergence (default 1.0 for standard VAE)
        """
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kld_loss


class ConditionalVAE(VAE):
    """Conditional VAE for label-guided clustering.
    
    Incorporates class labels into both encoder and decoder to learn
    class-aware latent representations.
    """
    
    def __init__(self, input_dim, n_classes, hidden_dim=256, latent_dim=32):
        """
        Arguments:
            input_dim: feature dimension
            n_classes: number of classes/clusters
            hidden_dim: hidden layer dimension
            latent_dim: latent space dimension
        """
        super().__init__(input_dim, hidden_dim, latent_dim)
        self.n_classes = n_classes
        
        # Class embedding
        self.embedding_dim = 32
        self.label_embedding = nn.Embedding(n_classes, self.embedding_dim)
        
        # Modified encoder that includes class information
        self.fc1_conditional = nn.Linear(input_dim + self.embedding_dim, hidden_dim)
        
        # Modified decoder that includes class information
        self.fc_decode_conditional = nn.Linear(latent_dim + self.embedding_dim, hidden_dim)
    
    def encode(self, x, y):
        """Encode with class condition.
        
        Arguments:
            x: input features (batch_size, input_dim)
            y: class labels (batch_size,)
        """
        y_embed = self.label_embedding(y)  # (batch, embedding_dim)
        x_y = torch.cat([x, y_embed], dim=1)
        h = F.relu(self.fc1_conditional(x_y))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def decode(self, z, y):
        """Decode with class condition.
        
        Arguments:
            z: latent vector (batch_size, latent_dim)
            y: class labels (batch_size,)
        """
        y_embed = self.label_embedding(y)
        z_y = torch.cat([z, y_embed], dim=1)
        h = F.relu(self.fc_decode_conditional(z_y))
        return self.fc_out(h)
    
    def forward(self, x, y):
        """Full CVAE forward pass.
        
        Arguments:
            x: input features
            y: class labels
        """
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, y)
        return recon, mu, logvar
    
    @staticmethod
    def loss_function(recon_x, x, mu, logvar, beta=1.0):
        """CVAE loss function (same as standard VAE)."""
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kld_loss