import torch
import sys
sys.path.append(sys.path[0] + '/..')
from config import device


class RefractiveIndex(float):
    VALUE_SPACE = [0.12, 0.306, 1.38, 1.45, 1.65, 1.8, 2.0, 2.15, 2.25]
    def __init__(self, value):
        assert value in self.VALUE_SPACE, "Specified value does not exist as refractive index"
        self.value = value

    def __str__(self):
        return str(self.value)

    @staticmethod
    def ceil(x: float):
        return RefractiveIndex(min(RefractiveIndex.VALUE_SPACE, key=lambda y: y if y >= x else float('inf')))

    @staticmethod
    def ceil_tensor(x: torch.Tensor):
        return torch.tensor([RefractiveIndex.ceil(val) for val in x.flatten()], device = device).view(x.shape)

    @staticmethod
    def floor(x: float):
        return RefractiveIndex(max(RefractiveIndex.VALUE_SPACE, key=lambda y: y if y <= x else float('-inf')))

    @staticmethod
    def floor_tensor(x: torch.Tensor):
        return torch.tensor([RefractiveIndex.floor(val) for val in x.flatten()], device = device).view(x.shape)


    @staticmethod
    def round(x: float):
        return RefractiveIndex(min(RefractiveIndex.VALUE_SPACE, key=lambda y: abs(y - x)))

    @staticmethod
    def round_tensor(x: torch.Tensor):
        return torch.tensor([RefractiveIndex.round(val) for val in x.flatten()], device = device).view(x.shape)
