from abc import ABC, abstractmethod
from torch.utils.data import Dataset, TensorDataset
from typing import Any, Optional, Union, List
import torch
import random
import math


class BaseDataloader(ABC):
    """
    Abstract base class for dataloaders.

    This class provides a common interface for loading data.
    It is intended to be subclassed by specific dataloaders with extended functionality.

    Attributes:
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset before each epoch.

    Methods:
        load_data: Load data into self.dataset. Must be implemented by subclasses.
    """

    def __init__(self, batch_size: int, shuffle: bool = True):
        """
        Initialise a BaseDataloader instance.

        Args:
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the dataset before each epoch. Defaults to True.
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset: Optional[Union[Dataset, List[torch.Tensor]]] = None

    @abstractmethod
    def load_data(self, *args, **kwargs):
        """Load data into self.dataset. Must be implemented by subclasses."""
        pass

    def __len__(self):
        """
        Return the number of batches in the dataset.

        Raises:
            ValueError: If the dataset is not loaded.
            TypeError: If the dataset type is unsupported.
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_data first.")

        if isinstance(self.dataset, Dataset):
            return math.ceil(len(self.dataset) / self.batch_size)
        elif isinstance(self.dataset, list):
            return math.ceil(len(self.dataset) / self.batch_size)
        else:
            raise TypeError("Unsupported dataset type.")

    def __getitem__(self, item):
        """
        Return the item at the given index.

        Args:
            item: The index of the item to return.

        Returns:
            The item at the given index.

        Raises:
            ValueError: If the dataset is not loaded.
            TypeError: If the dataset type is unsupported.
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_data first.")

        if isinstance(self.dataset, Dataset) or isinstance(self.dataset, list):
            return self.dataset[item]
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
