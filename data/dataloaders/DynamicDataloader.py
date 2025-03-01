from torch.utils.data import DataLoader
import torch
from typing import Callable
import sys
sys.path.append(sys.path[0] + '/../..')

from data.dataloaders.BaseDataloader import BaseDataloader

class DynamicDataloader(BaseDataloader):
    def __init__(self, batch_size: int, shuffle: bool = True, switch_condition: Callable[[int], bool] = None):
        super().__init__(batch_size, shuffle)
        self.dataloaders = []
        self.switch_condition = switch_condition
        self.current_index = 0
        self.epoch = 0

    def load_data(self, dataset_paths: list = []):
        for path in dataset_paths:
            dataset = torch.load("data/datasets/" + path)
            dataloader = DataLoader(dataset, batch_size = self.batch_size, shuffle = self.shuffle)
            self.dataloaders.append(dataloader)

    def next_epoch(self):
        """
        Call this method at the beginning of each epoch
        """
        if self.switch_condition is not None and self.switch_condition(self.epoch) and self.current_index < len(self.dataloaders) - 1:
            self.current_index += 1
        self.epoch += 1

    def __iter__(self):
        return iter(self.dataloaders[self.current_index])

    def __len__(self):
        return len(self.dataloaders[self.current_index])
