from abc import ABC, abstractmethod
import torch
import sys
sys.path.append(sys.path[0] + '/..')
from data.BaseDataloader import BaseDataloader
from data.values.ReflectiveProps import ReflectiveProps

class BasePredictionEngine(ABC):

    def __init__(self, dataloader: BaseDataloader):
        self.dataloader = dataloader

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self, reflective_props: ReflectiveProps):
        pass