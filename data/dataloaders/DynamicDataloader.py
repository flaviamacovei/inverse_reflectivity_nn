from torch.utils.data import DataLoader
import torch
from typing import Callable
import sys
sys.path.append(sys.path[0] + '/../..')

from data.dataloaders.BaseDataloader import BaseDataloader
from utils.ConfigManager import ConfigManager as CM
from data.material_embedding.EmbeddingManager import EmbeddingManager as EM

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
        for name in dataset_paths:
            try:
                filepath = "data/datasets/" + name + f"_{EM().hash_materials()}.pt"
                dataset = torch.load(filepath)
                self.datasets.extend(dataset)
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
