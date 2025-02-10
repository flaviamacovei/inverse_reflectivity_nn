from abc import ABC, abstractmethod
from BasePredictionEngine import BasePredictionEngine
import sys
sys.path.append(sys.path[0] + '/..')
from data.BaseDataloader import BaseDataloader

class BaseTrainablePredictionEngine(BasePredictionEngine, ABC):
    def __init__(self, dataloader: BaseDataloader):
        super().__init__()
        self.dataloader = dataloader

    @abstractmethod
    def train(self):
        pass