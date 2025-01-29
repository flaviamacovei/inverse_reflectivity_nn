from abc import ABC, abstractmethod
from typing import Any, Iterator
import math


class BaseDataLoader(ABC):
    def __init__(self, batch_size: int, shuffle: bool = False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset: Any = None
        self.index = 0  # keep track of iteration position
        self.indices = []  # store shuffled indices if shuffling enabled

    @abstractmethod
    def load_data(self):
        """
        Abstract method that populates self.dataset with data
        """
        pass

    def __iter__(self) -> Iterator[Any]:
        """
        Reset index and shuffle indices if needed before iteration
        """
        self.index = 0
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")

        self.indices = list(range(len(self.dataset)))
        if self.shuffle:
            import random
            random.shuffle(self.indices)
        return self

    def __next__(self) -> Any:
        """
        Return next batch of data
        """
        if self.index >= len(self.dataset):
            raise StopIteration

        batch_indices = self.indices[self.index:self.index + self.batch_size]
        batch = [self.dataset[i] for i in batch_indices]
        self.index += self.batch_size
        return batch

    def __len__(self) -> int:
        """
        Returns number of batches in dataset
        """
        return math.ceil(len(self.dataset) / self.batch_size)
