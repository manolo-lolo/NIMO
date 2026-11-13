import os
import torch
from torch.utils.data import Dataset
import pandas as pd


class SyntheticDataset(Dataset):
    def __init__(self, data_path, split="train", transform=None):
        self.data_path = data_path
        self.split = split
        assert self.split in ["train", "val", "test"], (
            "split should be either 'train', 'val' or 'test'"
        )

        self.transform = transform

        self.data = pd.read_csv(os.path.join(self.data_path, f"{self.split}.csv"))

        self.target = torch.tensor(self.data["target"].values, dtype=torch.float32)
        self.features = torch.tensor(
            self.data.drop(columns=["target"]).values, dtype=torch.float32
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = self.features[idx]
        y = self.target[idx]

        if self.transform:
            X = self.transform(X)

        sample = {"features": X, "target": y}
        return sample
