import torch
from typing import Union
import sys
sys.path.append(sys.path[0] + '/..')
from utils.ConfigManager import ConfigManager as CM

class Material:
    def __init__(self, title: str, B: Union[torch.tensor, list[float]], C: Union[torch.tensor, list[float]]):
        assert len(B) == len(C)
        assert len(B) > 0
        self.title = title
        self.B = B if isinstance(B, torch.Tensor) else torch.tensor(B, device = CM().get('device'))
        self.C = C if isinstance(C, torch.Tensor) else torch.tensor(C, device = CM().get('device'))

    def get_refractive_indices(self):
        wl2 = CM().get('wavelengths') ** 2
        n2 = 1 + sum(b * wl2 / (wl2 - c) for b, c in zip(self.B, self.C))
        return n2 ** 0.5

    def get_B(self):
        return self.B

    def get_C(self):
        return self.C

    def get_coeffs(self):
        return torch.cat((self.get_B(), self.get_C()))

    def get_title(self):
        return self.title

    def __repr__(self):
        return self.title

    def __str__(self):
        return f"{self.title}:\nB: {self.B.cpu().detach().numpy()}\nC: {self.C.cpu().detach().numpy()}"

    def __eq__(self, other):
        return self.title == other.get_title() and torch.all(self.B == other.get_B()) and torch.all(self.C == other.get_C())
