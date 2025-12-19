import os
import numpy as np
import torch
from torch.utils.data import Dataset

class FeatureDataset(Dataset):
    """Loads per-track fixed-size feature vectors saved as .npy files."""
    def __init__(self, feat_dir, transform=None):
        self.files = sorted([os.path.join(feat_dir, f) for f in os.listdir(feat_dir) if f.endswith('.npy')])
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x = np.load(self.files[idx])
        x = x.astype('float32')
        if self.transform:
            x = self.transform(x)
        return torch.from_numpy(x)
