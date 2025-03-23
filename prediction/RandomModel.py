from prediction.BaseModel import BaseModel
import torch
import sys
sys.path.append(sys.path[0] + '/..')
from data.values.ReflectivePropsPattern import ReflectivePropsPattern
from data.values.Coating import Coating

class RandomModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.num_layers = CM().get('num_layers')

    def predict(self, target: ReflectivePropsPattern):
        thicknesses_tensor = torch.rand((self.num_layers))
        refractive_indices_tensor = torch.rand((self.num_layers))
        return Coating(thicknesses_tensor, refractive_indices_tensor)