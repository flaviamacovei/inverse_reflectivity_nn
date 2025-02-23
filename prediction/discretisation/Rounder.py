import torch
import sys
sys.path.append(sys.path[0] + '/..')
from prediction.relaxation.BaseRelaxedSolver import BaseRelaxedSolver
from prediction.discretisation.BaseDiscretiser import BaseDiscretiser
from data.values.ReflectivePropsPattern import ReflectivePropsPattern
from data.values.Coating import Coating
from data.values.RefractiveIndex import RefractiveIndex
from config import device
from ui.visualise import visualise
from forward.forward_tmm import coating_to_reflective_props

class Rounder(BaseDiscretiser):
    def __init__(self, relaxed_solver: BaseRelaxedSolver):
        super().__init__(relaxed_solver)

    def solve_discretised(self, target: ReflectivePropsPattern):
        relaxed_prediction = self.relaxed_solver.solve_relaxed(target)
        relaxed_prediction_value = coating_to_reflective_props(relaxed_prediction)
        visualise(preds = relaxed_prediction_value, filename = "before_rounding")
        return self.round(relaxed_prediction)

    def round(self, coating: Coating):
        thicknesses = coating.get_thicknesses()
        refractive_indices = coating.get_refractive_indices()
        return Coating(thicknesses, RefractiveIndex.round_tensor(refractive_indices))
