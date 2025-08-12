import numpy as np
import torch
from torch.utils.data import Dataset

class Slices2D(Dataset):
    def __init__(self, images_dir, labels_dir, size=256, k_slices=1,
                 split='train', split_ratio=(0.8,0.1,0.1), seed=0):
        # TODO: replace with real NIfTI loading + splitting
        self.size=size; self.k=k_slices; self.n=32
    def __len__(self): return self.n
    def __getitem__(self, i):
        x = torch.zeros(self.k, self.size, self.size)
        y = torch.zeros(1, self.size, self.size)
        return x, y
