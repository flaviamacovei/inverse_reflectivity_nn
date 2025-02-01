from prediction.BasePredictionEngine import BasePredictionEngine
import torch
import sys
sys.path.append(sys.path[0] + '/..')
from data.BaseDataloader import BaseDataloader

class RandomPredictor(BasePredictionEngine):
    def __init__(self, dataloader: BaseDataloader):
        super().__init__(dataloader = dataloader)

    def train(self):
        pass

    def predict(self, properties: torch.Tensor):
        return torch.rand((4))