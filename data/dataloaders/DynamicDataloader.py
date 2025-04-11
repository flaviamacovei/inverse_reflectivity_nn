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
        self.dataloaders = []
        self.datasets = []
        switch_conditions = {
            True: lambda epoch: epoch % max(1, CM().get('training.num_epochs') // (len(CM().get('dataset_files')) + 1)) == 0,
            False: lambda epoch: False
        }
        self.switch_condition = switch_conditions[CM().get('training.dataset_switching')]
        self.current_index = 0
        self.epoch = 0

    def load_data(self, dataset_paths: list = []):
        guidace = CM().get('training.guidance')
        if CM().get('training.dataset_switching'):
            if guidace == "free":
                densities = ["complete", "masked", "explicit"]
            elif guidace == "guided":
                densities = ["complete", "masked"]
            else:
                raise ValueError(f"Guidance {guidace} not supported.")
        else:
            densities = ["complete"]
        for density in densities:
            try:
                filepath = get_dataset_name("training", density)
                if os.path.exists(filepath):
                    dataset = torch.load(filepath)
                else:
                    segment_files = []
                    i = 1
                    while os.path.exists(filepath[:-3] + f"_seg_{i}.pt"):
                        segment_files.append(filepath[:-3] + f"_seg_{i}.pt")
                        i += 1
                    dataset = SegmentedDataset(segment_files)
                self.datasets.extend(dataset)
            except AttributeError:
                raise FileNotFoundError("Dataset in current configuration not found. Please run generate_dataset.py first.")
            except FileNotFoundError:
                raise FileNotFoundError("Dataset in current configuration not found. Please run generate_dataset.py first.")
            dataloader = DataLoader(dataset, batch_size = self.batch_size, shuffle = self.shuffle, drop_last = True)
            self.dataloaders.append(dataloader)

    def next_epoch(self):
        """
        Call this method at the beginning of each epoch
        """
        self.epoch += 1
        if self.switch_condition(self.epoch) and self.current_index < len(self.dataloaders) - 1:
            self.current_index += 1
            print("On dataloader " + str(self.current_index))

    def __getitem__(self, item):
        try:
            return self.datasets[item]
        except IndexError:
            raise IndexError("Index out of range")


    def __iter__(self):
        return iter(self.dataloaders[self.current_index])

    def __len__(self):
        return len(self.datasets)
