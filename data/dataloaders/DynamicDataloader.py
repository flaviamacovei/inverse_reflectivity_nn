from torch.utils.data import DataLoader, Dataset
import torch
from typing import Callable
import os
import sys
sys.path.append(sys.path[0] + '/../..')

from data.dataloaders.BaseDataloader import BaseDataloader
from utils.ConfigManager import ConfigManager as CM
from data.material_embedding.EmbeddingManager import EmbeddingManager as EM
from utils.data_utils import get_dataset_name

class SegmentedDataset(Dataset):
    """
    Dataset class for loading data from multiple files (segments).

    Attributes:
        segment_files: List of segment file paths.
        segments: List of loaded segments as datasets
        num_samples: Total number of samples in the dataset.

    Methods:
        load_segments: Load data from each segment file.
    """

    def __init__(self, segment_files: list):
        """
        Initialise a SegmentedDataset instance.

        Args:
            segment_files: List of segment file paths.
        """
        self.segment_files = segment_files
        self.segments = []
        self.num_samples = self.load_segments()

    def load_segments(self):
        """Load data from each segment file and store in self.segments."""
        total_samples = 0
        for file in self.segment_files:
            if os.path.exists(file):
                segment_data = torch.load(file)
                self.segments.append(segment_data)
                total_samples += len(segment_data)
        return total_samples

    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, index):
        """Return the sample at the given index."""
        segment_index = 0
        while index >= len(self.segments[segment_index]):
            index -= len(self.segments[segment_index])
            segment_index += 1
        return self.segments[segment_index][index]

class DynamicDataloader(BaseDataloader):
    """
    Dynamic Dataloader class for automatic handling of different datasets based on config settings.

    Attributes:
        dataset: The loaded dataset.
        dataloader: The DataLoader for the dataset.

    Methods:
        load_data: Load data from the specified dataset file.
        load_leg: Load configuration for a specific leg.
    """

    def __init__(self, batch_size: int, shuffle: bool = True):
        """
        Initialise a DynamicDataloader instance.

        Args:
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle the dataset before each epoch. Defaults to True.
        """
        super().__init__(batch_size, shuffle)
        self.dataset = None

    def load_leg(self, leg: int = 0):
        """
        Load configuration for a specific leg.

        Args:
            leg: Index of the leg from which to load configuration.
        """
        density = CM().get(f'training.guidance_schedule.{leg}.density')
        filepath = get_dataset_name("training", density)
        self.load_data(filepath)
        # TODO: turn this into an actual dataloader class maybe?
        self.dataloader = DataLoader(self.dataset, batch_size = self.batch_size, shuffle = self.shuffle)

    def load_data(self, filepath: str = None):
        """
        Load data from the specified dataset file.

        Args:
            filepath: Path to the dataset file.

        Raises:
            FileNotFoundError: If the dataset file is not found.
            AttributeError: If the dataset file is not found.
        """
        try:
            if filepath is None:
                raise FileNotFoundError(
                    "Dataset in current configuration not found. Please run generate_dataset.py first.")

            if os.path.exists(filepath):
                # filepath is a single file, dataset not segmented
                self.dataset = torch.load(filepath)
            else:
                # filepath is a prefix, dataset is segmented
                segment_files = []
                i = 1
                while os.path.exists(filepath[:-3] + f"_seg_{i}.pt"):
                    segment_files.append(filepath[:-3] + f"_seg_{i}.pt")
                    i += 1
                self.dataset = SegmentedDataset(segment_files)
        except AttributeError:
            raise FileNotFoundError("Dataset in current configuration not found. Please run generate_dataset.py first.")
        except FileNotFoundError:
            raise FileNotFoundError("Dataset in current configuration not found. Please run generate_dataset.py first.")


    def __getitem__(self, item):
        """
        Return the item at the given index.

        Args:
            item: The index of the item to return.

        Returns:
            The item at the given index.

        Raises:
            IndexError: If the index is out of range.
        """
        try:
            return self.dataset[item]
        except IndexError:
            raise IndexError("Index out of range")


    def __iter__(self):
        """Return an iterator for the dataset."""
        return iter(self.dataloader)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.batch_indices)
