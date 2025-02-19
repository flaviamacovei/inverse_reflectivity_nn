import torch
import sys
sys.path.append(sys.path[0] + '/..')
from prediction.relaxation.BaseRelaxedSolver import BaseRelaxedSolver
from prediction.discretisation.BaseDiscretiser import BaseDiscretiser
from data.values.ReflectivePropsPattern import ReflectivePropsPattern
from data.values.Coating import Coating
from data.values.RefractiveIndex import RefractiveIndex

class Rounder(BaseDiscretiser):
    def __init__(self, relaxed_solver: BaseRelaxedSolver):
        super().__init__(relaxed_solver)

    def solve_discretised(self, target: ReflectivePropsPattern):
        relaxed_prediction = self.relaxed_solver.solve_relaxed(target)
        return self.round(relaxed_prediction)

    def round(self, coating: Coating):
        thicknesses = coating.get_thicknesses()
        refractive_indices = coating.get_refractive_indices()
        refractive_indices_rounded = torch.tensor([RefractiveIndex.round(x) for x in refractive_indices])
        return Coating(thicknesses, refractive_indices_rounded)
