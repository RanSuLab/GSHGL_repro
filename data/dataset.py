"""
WSI graph dataset loader.

- Loads per-slide graph ``.pt`` files produced by ``preprocess.py``
- Supports reading graphs from a directory
- Also compatible with cached graph lists
"""

import os
from pathlib import Path
from torch_geometric.data import Dataset, Data
import torch


class WsiGraphDataset(Dataset):
    """
    Dataset wrapper for per-slide graph files.

    Args:
        root: Directory containing graph files.

    Each ``.pt`` file in the directory should be a PyG ``Data`` object.
    """
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.pt_files = sorted([p for p in Path(root).glob("*.pt")])
        if len(self.pt_files) == 0:
            raise RuntimeError(f"No .pt graph files found in {root}; run preprocess.py first")

    def len(self):
        return len(self.pt_files)

    def get(self, idx):
        data = torch.load(self.pt_files[idx])
        return data
