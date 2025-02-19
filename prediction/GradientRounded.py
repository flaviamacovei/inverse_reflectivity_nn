import sys
sys.path.append(sys.path[0] + '/..')
from prediction.BasePredictionEngine import BasePredictionEngine
from prediction.relaxation.GradientSolver import GradientSolver
from prediction.discretisation.Rounder import Rounder

class GradientRounded(BasePredictionEngine):

    def __init__(self, num_layers: int):
        self.relaxed_solver = GradientSolver(num_layers)
        self.discretiser = Rounder(self.relaxed_solver)