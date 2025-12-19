import os
import numpy as np
import torch
from torch.utils.data import Dataset


class SpectrogramDataset(Dataset):
    """Loads per-track spectrogram .npy files saved as 2D arrays (n_mels x frames)."""
    def __init__(self, spec_dir, transform=None):
        self.files = sorted([os.path.join(spec_dir, f) for f in os.listdir(spec_dir) if f.endswith('.npy')])
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = np.load(self.files[idx])
        # Add channel dim
        x = np.expand_dims(x, 0).astype('float32')
        if self.transform:
            x = self.transform(x)
        return torch.from_numpy(x)
