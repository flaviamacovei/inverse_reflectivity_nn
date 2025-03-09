import torch
import sys
sys.path.append(sys.path[0] + '/..')
from utils.ConfigManager import ConfigManager as CM

class Material:
    def __init__(self, title: str, B: list[float], C: list[float]):
        assert len(B) > 0
        assert len(B) == len(C)
        self.title = title
        self.B = B
        self.C = C

    def get_refractive_indices(self):
        wl2 = CM().get('wavelengths') ** 2
        n2 = 1 + sum(b * wl2 / (wl2 - c) for b, c in zip(self.B, self.C))
        return n2 ** 0.5

    def get_B(self):
        return torch.tensor(self.B)

    def get_C(self):
        return torch.tensor(self.C)

    def get_title(self):
        return self.title

    def __repr__(self):
        return self.title

    def __str__(self):
        return f"{self.title}:\nB: {self.B}\nC: {self.C}"
