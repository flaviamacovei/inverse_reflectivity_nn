from abc import ABC, abstractmethod
from torch.utils.data import Dataset, TensorDataset
from typing import Any, Optional, Union, List
import torch
import random


class BaseDataloader(ABC):
    def __init__(self, batch_size: int, shuffle: bool = True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset: Optional[Union[Dataset, List[torch.Tensor]]] = None

    @abstractmethod
    def load_data(self, *args, **kwargs):
        """Load data into self.dataset. Must be implemented by subclasses."""
        pass

    def __len__(self):
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_data first.")

        if isinstance(self.dataset, Dataset):
            return len(self.dataset) // self.batch_size
        elif isinstance(self.dataset, list):
            return len(self.dataset) // self.batch_size
        else:
            raise TypeError("Unsupported dataset type.")

    def __getitem__(self, index: int):
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_data first.")

        if isinstance(self.dataset, Dataset):
            return self.dataset[index * self.batch_size: (index + 1) * self.batch_size]
        elif isinstance(self.dataset, list):
            return self.dataset[index * self.batch_size: (index + 1) * self.batch_size]
        else:
            raise TypeError("Unsupported dataset type.")

    def get_batches(self):
        """Generator function that yields batches from the dataset."""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_data first.")

        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i: i + self.batch_size]
            yield [self.dataset[idx] for idx in batch_indices]
