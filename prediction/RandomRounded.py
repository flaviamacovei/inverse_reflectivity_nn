import sys
sys.path.append(sys.path[0] + '/..')
from prediction.BasePredictionEngine import BasePredictionEngine
from prediction.relaxation.RandomSolver import RandomSolver
from prediction.discretisation.Rounder import Rounder

class RandomRounded(BasePredictionEngine):

    def __init__(self, num_layers: int):
        self.relaxed_solver = RandomSolver(num_layers)
        self.discretiser = Rounder(self.relaxed_solver)