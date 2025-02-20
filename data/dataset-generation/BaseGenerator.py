from abc import ABC, abstractmethod
import random
import torch
import sys
sys.path.append(sys.path[0] + '/../..')
from data.values.Coating import Coating
from data.values.RefractiveIndex import RefractiveIndex
from config import tolerance

class BaseGenerator(ABC):
    def __init__(self, num_points: int):
        self.num_points = num_points
        self.TOLERANCE = tolerance

    def make_random_coating(self):
        num_layers = random.randint(3, 13)
        thicknesses = torch.rand((num_layers))
        thicknesses[0] = float("Inf")
        thicknesses[-1] = float("Inf")
        refractive_indices = torch.rand(( num_layers))
        refractive_indices_rounded = torch.tensor([RefractiveIndex.round(x) for x in refractive_indices])
        return Coating(thicknesses, refractive_indices_rounded)

    @abstractmethod
    def make_point(self):
        pass

    def generate(self):
        points = []
        for _ in range(self.num_points):
            points.append(self.make_point())
        return points
