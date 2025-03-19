import torch
from typing import Union
import sys
sys.path.append(sys.path[0] + '/..')
from data.values.Material import Material
from utils.ConfigManager import ConfigManager as CM

class SellmeierMaterial(Material):
    def __init__(self, title: str, B: Union[torch.tensor, list[float]], C: Union[torch.tensor, list[float]]):
        assert len(B) == len(C)
        assert len(B) > 0
        super().__init__(title)
        self.B = B if isinstance(B, torch.Tensor) else torch.tensor(B, device = CM().get('device'))
        self.C = C if isinstance(C, torch.Tensor) else torch.tensor(C, device = CM().get('device'))

    def get_refractive_indices(self):
        wl2 = CM().get('wavelengths') ** 2
        n2 = 1 + sum(b * wl2 / (wl2 - c) for b, c in zip(self.B, self.C))
        return n2 ** 0.5

    def get_coeffs(self):
        return torch.cat((self.get_B(), self.get_C()))

    def __str__(self):
        return f"{self.title}:\nB: {self.B.cpu().detach().numpy()}\nC: {self.C.cpu().detach().numpy()}"

    def __hash__(self):
        return abs(torch.cdist(self.B[None, :], self.C[None, :], p = 2).sum().item() * 5734)
