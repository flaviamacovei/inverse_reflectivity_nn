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
    def __init__(self, segment_files: list):
        self.segment_files = segment_files
        self.segments = []
        self.num_samples = self.load_segments()

    def load_segments(self):
        total_samples = 0
        for file in self.segment_files:
            if os.path.exists(file):
                segment_data = torch.load(file)
                self.segments.append(segment_data)
                total_samples += len(segment_data)
        return total_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        segment_index = 0
        while index >= len(self.segments[segment_index]):
            index -= len(self.segments[segment_index])
            segment_index += 1
        return self.segments[segment_index][index]

class DynamicDataloader(BaseDataloader):
    def __init__(self, batch_size: int, shuffle: bool = True):
        super().__init__(batch_size, shuffle)
        self.dataset = None

    def load_leg(self, leg: int = 0):
        density = CM().get(f'training.guidance_schedule.{leg}.density')
        filepath = get_dataset_name("training", density)
        self.load_data(filepath)
        # TODO: turn this into an actual dataloader class maybe?
        self.dataloader = DataLoader(self.dataset, batch_size = self.batch_size, shuffle = self.shuffle)

    def load_data(self, filepath: str = None):
        try:
            if filepath is None:
                raise FileNotFoundError(
                    "Dataset in current configuration not found. Please run generate_dataset.py first.")

            if os.path.exists(filepath):
                self.dataset = torch.load(filepath)
            else:
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
        try:
            return self.dataset[item]
        except IndexError:
            raise IndexError("Index out of range")


    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.batch_indices)
