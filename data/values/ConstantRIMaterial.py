import torch
from typing import Union
import sys
sys.path.append(sys.path[0] + '/..')
from data.values.Material import Material
from utils.ConfigManager import ConfigManager as CM

class ConstantRIMaterial(Material):
    def __init__(self, title: str, reflective_index: Union[torch.Tensor, float]):
        super().__init__(title)
        self.refractive_index = reflective_index if isinstance(reflective_index, torch.Tensor) else torch.tensor([reflective_index], device = CM().get('device'))

    def get_refractive_indices(self):
        result = torch.zeros_like(CM().get('wavelengths'))
        result[:] = self.refractive_index
        return result

    def get_coeffs(self):
        return self.refractive_index

    def __str__(self):
        return f"{self.title}:\nrefractive index: {self.refractive_index.cpu().detach().numpy()}"

    def __hash__(self):
        return abs(int(self.refractive_index.item() * 5407))