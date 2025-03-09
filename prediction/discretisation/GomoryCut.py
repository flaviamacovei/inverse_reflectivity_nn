import random
import sys
sys.path.append(sys.path[0] + '/..')
from prediction.relaxation.BaseRelaxedSolver import BaseRelaxedSolver
from prediction.discretisation.BaseDiscretiser import BaseDiscretiser
from data.values.ReflectivePropsPattern import ReflectivePropsPattern
from data.values.RefractiveIndex import RefractiveIndex
from data.values.Coating import Coating

class GomoryCut(BaseDiscretiser):
    def __init__(self, relaxed_solver: BaseRelaxedSolver):
        super().__init__(relaxed_solver)

    def predict(self, target: ReflectivePropsPattern):
        relaxed_solution = self.relaxed_solver.solve(target)
        rounded_solution = Coating(relaxed_solution.get_thicknesses(), RefractiveIndex.round_tensor(relaxed_solution.get_refractive_indices()))
        if relaxed_solution != rounded_solution:
            nondiscrete_indices = self.get_nondiscrete_map(relaxed_solution.get_refractive_indices())
            round_index = random.choice(nondiscrete_indices)
            self.relaxed_solver.increment_output_size()
            new_relaxed_solution = self.relaxed_solver.solve(target)
            print(f"new solution shape: {new_relaxed_solution.get_thicknesses().shape}, {new_relaxed_solution.get_refractive_indices().shape}")
            new_rounded_solution = Coating(new_relaxed_solution.get_thicknesses(), RefractiveIndex.round_tensor(new_relaxed_solution.get_refractive_indices()))
            new_nondiscrete_indices = RefractiveIndex.get_nondiscrete_indices(new_relaxed_solution.get_refractive_indices())
            print(f"new nondiscrete indices: {new_nondiscrete_indices}")
            print("unequal")
        return rounded_solution
