from abc import ABC, abstractmethod
import torch
import sys
sys.path.append(sys.path[0] + '/..')
from data.DataloaderInterface import BaseDataloader

class BasePredictionEngine(ABC):

    def __init__(self, dataloader: BaseDataloader):
        self.dataloader = dataloader

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self, properties: torch.Tensor):
        pass