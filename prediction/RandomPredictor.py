from prediction.BasePredictionEngine import BasePredictionEngine
import torch
import sys
sys.path.append(sys.path[0] + '/..')
from data.BaseDataloader import BaseDataloader
from data.values.ReflectiveProps import ReflectiveProps
from data.values.Coating import Coating

class RandomPredictor(BasePredictionEngine):
    def __init__(self, dataloader: BaseDataloader, num_layers: int):
        super().__init__(dataloader = dataloader)
        self.num_layers = num_layers

    def train(self):
        pass

    def predict(self, reflecive_props: ReflectiveProps):
        thicknesses_tensor = torch.rand((self.num_layers))
        return Coating(thicknesses_tensor)