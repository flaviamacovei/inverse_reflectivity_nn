from abc import ABC, abstractmethod
import random
import torch
import sys
sys.path.append(sys.path[0] + '/../..')
from data.values.Coating import Coating
from data.values.RefractiveIndex import RefractiveIndex
from utils.ConfigManager import ConfigManager as CM

class BaseGenerator(ABC):
    def __init__(self, num_points: int):
        self.num_points = num_points
        self.TOLERANCE = CM().get('tolerance')

    def make_random_coating(self):
        num_layers = random.randint(CM().get('layers.min'), CM().get('layers.max'))
        thicknesses = torch.rand((num_layers), device = CM().get('device')) / 1.0e6
        # thicknesses[0] = 10_000#float("Inf")
        # thicknesses[-1] = 10_000#float("Inf")
        refractive_indices = (2.25 - 0.12) * torch.rand((num_layers)) + 0.12
        refractive_indices_rounded = torch.tensor([RefractiveIndex.round(x) for x in refractive_indices], device = CM().get('device'))
        return Coating(thicknesses, refractive_indices_rounded)

    @abstractmethod
    def make_point(self):
        pass

    def generate(self):
        points = []
        for i in range(self.num_points):
            if i % (max(self.num_points // 10, 1)) == 0:
                print(f"{i}/{self.num_points}")
            points.append(self.make_point())
        return points
