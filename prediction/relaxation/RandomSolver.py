from prediction.relaxation.BaseRelaxedSolver import BaseRelaxedSolver
import torch
import sys
sys.path.append(sys.path[0] + '/..')
from data.values.ReflectivePropsPattern import ReflectivePropsPattern
from data.values.Coating import Coating

class RandomSolver(BaseRelaxedSolver):
    def __init__(self, num_layers: int = 19):
        super().__init__()
        self.num_layers = num_layers

    def solve(self, target: ReflectivePropsPattern):
        thicknesses_tensor = torch.rand((self.num_layers))
        refractive_indices_tensor = torch.rand((self.num_layers))
        return Coating(thicknesses_tensor, refractive_indices_tensor)