import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    """A simple convolutional VAE for fixed-size log-mel spectrogram inputs.

    Input shape: (batch, 1, n_mels, n_frames)
    """
    def __init__(self, in_channels=1, hidden_channels=32, latent_dim=32, kernel_size=4):
        super().__init__()
        # store kernel size for flexible conv shapes
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        # Encoder
        self.enc1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size, stride=2, padding=padding)  # -> /2
        self.enc2 = nn.Conv2d(hidden_channels, hidden_channels*2, kernel_size=kernel_size, stride=2, padding=padding)  # -> /4
        self.enc3 = nn.Conv2d(hidden_channels*2, hidden_channels*4, kernel_size=kernel_size, stride=2, padding=padding)  # -> /8

        self._enc_out_dim = None  # computed later

        # store hidden channels for decoder construction
        self.hidden_channels = hidden_channels

        # bottleneck linear layers (set after an input is seen)
        self.fc_mu = None
        self.fc_logvar = None
        self.fc_dec = None

        # decoder convs (constructed lazily)
        self.dec_conv1 = None
        self.dec_conv2 = None
        self.dec_conv3 = None

        self.latent_dim = latent_dim

    def _init_fc_layers(self, x_shape, device=None):
        # x_shape after convs: (C, H, W)
        C, H, W = x_shape
        flat = C * H * W
        self._enc_out_dim = flat
        self.fc_mu = nn.Linear(flat, self.latent_dim).to(device)
        self.fc_logvar = nn.Linear(flat, self.latent_dim).to(device)
        self.fc_dec = nn.Linear(self.latent_dim, flat).to(device)
        # decoder conv transpose layers constructed based on stored hidden_channels
        hc = self.hidden_channels
        # C should equal hc*4 from encoder
        k = self.kernel_size
        p = k // 2
        self.dec_conv1 = nn.ConvTranspose2d(C, hc*2, kernel_size=k, stride=2, padding=p).to(device)
        self.dec_conv2 = nn.ConvTranspose2d(hc*2, hc, kernel_size=k, stride=2, padding=p).to(device)
        self.dec_conv3 = nn.ConvTranspose2d(hc, 1, kernel_size=k, stride=2, padding=p).to(device)

    def encode(self, x):
        h = F.relu(self.enc1(x))
        h = F.relu(self.enc2(h))
        h = F.relu(self.enc3(h))
        # store shape for decoder
        b, c, h_, w_ = h.shape
        if self._enc_out_dim is None:
            # lazy init of fc layers aligned with device
            self._init_fc_layers((c, h_, w_), device=h.device)
        h_flat = h.view(h.size(0), -1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        return mu, logvar, (c, h_, w_)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, shape):
        b = z.size(0)
        flat = self.fc_dec(z)
        c, h_, w_ = shape
        h = flat.view(b, c, h_, w_)
        # mirror encoder convs using convtranspose
        # simple upsampling via interpolate + conv
        # decode using transpose convs
        h = F.relu(self.dec_conv1(h))
        h = F.relu(self.dec_conv2(h))
        h = self.dec_conv3(h)
        return h

    def forward(self, x):
        mu, logvar, shape = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, shape)
        return recon, mu, logvar

    @staticmethod
    def loss_function(recon_x, x, mu, logvar, beta=1.0):
        # If shapes mismatch due to kernel/padding choices or small input dims,
        # safely resize reconstruction to match target before loss computation.
        if recon_x.shape != x.shape:
            try:
                recon_x = F.interpolate(recon_x, size=x.shape[2:], mode='bilinear', align_corners=False)
            except Exception:
                # fallback: center-crop or pad to match
                rx_h, rx_w = recon_x.shape[2], recon_x.shape[3]
                tx_h, tx_w = x.shape[2], x.shape[3]
                min_h, min_w = min(rx_h, tx_h), min(rx_w, tx_w)
                recon_x = recon_x[..., :min_h, :min_w]
                x = x[..., :min_h, :min_w]
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kld_loss
