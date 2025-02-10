from abc import ABC, abstractmethod
import torch
import sys
sys.path.append(sys.path[0] + '/../..')
from data.values.Coating import Coating
from config import tolerance

class BaseGenerator(ABC):
    def __init__(self, num_points: int):
        self.num_points = num_points
        self.NUM_LAYERS = 19
        self.START_WL = 500
        self.END_WL = 1500
        self.STEPS = 1000
        self.TOLERANCE = tolerance

    def make_random_coating(self):
        thicknesses = torch.rand((1, self.NUM_LAYERS))
        thicknesses[:, 0] = float("Inf")
        thicknesses[:, -1] = float("Inf")
        return Coating(thicknesses)

    @abstractmethod
    def make_point(self):
        pass

    def generate(self):
        points = []
        for _ in range(self.num_points):
            points.append(self.make_point())
        return points
