import json
import numpy as np
import torch
from torch.utils.data import Dataset

class AudioTokenDataset(Dataset):

    def __init__(self, split: str, data_path: str, metadata_path: str):
        self.data_path = data_path
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        self.num_samples = metadata[split]["num_samples"]
        self.tokens_per_sample = metadata[split]["tokens_per_sample"]
        self.dtype = np.dtype(metadata[split]["dtype"])
        self.data = None

    def _open(self) -> None:
        if self.data is None:
            self.data = np.memmap(
                self.data_path,
                dtype=self.dtype,
                mode="r",
                shape=(self.num_samples, self.tokens_per_sample),
            )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        self._open()
        assert isinstance(self.data, np.memmap)

        row = self.data[idx]
        tokens = torch.from_numpy(row.copy()).long()

        return tokens