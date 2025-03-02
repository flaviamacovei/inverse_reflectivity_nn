import random
import torch
import sys
sys.path.append(sys.path[0] + '/..')
from prediction.discretisation.BaseDiscretiser import BaseDiscretiser
from prediction.relaxation.BaseRelaxedSolver import BaseRelaxedSolver
from data.values.ReflectivePropsPattern import ReflectivePropsPattern
from data.values.TreeNode import TreeNode
from forward.forward_tmm import coating_to_reflective_props
from evaluation.loss import match
from data.values.RefractiveIndex import RefractiveIndex
from data.values.Coating import Coating
from ui.visualise import visualise

class BranchAndBound(BaseDiscretiser):
    def __init__(self, relaxed_solver: BaseRelaxedSolver):
        super().__init__(relaxed_solver)

    def solve(self, target: ReflectivePropsPattern):
        solution_tree = [self.make_node(target, self.relaxed_solver.get_lower_bound(), self.relaxed_solver.get_upper_bound())]
        value = coating_to_reflective_props(solution_tree[0].get_coating())
        visualise(value, target, "before_rounding")
        max_error = float("Inf")
        optimum = None
        max_iter = 300
        while len(solution_tree) > 0:
            if max_iter % (max(max_iter // 10, 1)) == 0:
                print(max_iter)

            current_node = solution_tree.pop()
            refractive_indices = current_node.get_coating().get_refractive_indices()
            if current_node.get_error() < max_error:
                if RefractiveIndex.is_discrete(refractive_indices):
                    optimum = current_node.get_coating()
                    max_error = current_node.get_error()
                else:
                    offset = current_node.get_coating().get_thicknesses().shape[1]
                    branching_index = self.select_branching_index(refractive_indices)
                    offending_value = current_node.get_coating().get_refractive_indices()[:, branching_index]
                    branching_index = branching_index + offset

                    lower_bound = current_node.get_lower_bound()
                    upper_bound = current_node.get_upper_bound()

                    upper_bound_floor = upper_bound.clone()
                    upper_bound_floor[:, branching_index] = RefractiveIndex.floor_tensor(offending_value)
                    floor_node = self.make_node(target, lower_bound, upper_bound_floor)
                    solution_tree.append(floor_node)

                    lower_bound_ceil = lower_bound.clone()
                    lower_bound_ceil[:, branching_index] = RefractiveIndex.ceil_tensor(offending_value)
                    ceil_node = self.make_node(target, lower_bound_ceil, upper_bound)
                    solution_tree.append(ceil_node)

            max_iter -= 1
            if max_iter == 0:
                if optimum:
                    print("Maximum iterations reached, returning current optimum")
                    return optimum
                else:
                    print("Maximum iterations reached, resorting to rounding")
                    thicknesses = current_node.get_coating().get_thicknesses()
                    refractive_indices = current_node.get_coating().get_refractive_indices()
                    return Coating(thicknesses, RefractiveIndex.round_tensor(refractive_indices))

        return optimum

    def make_node(self, target: ReflectivePropsPattern, lower_bound: torch.Tensor, upper_bound: torch.Tensor):
        self.relaxed_solver.set_lower_bound(lower_bound)
        self.relaxed_solver.set_upper_bound(upper_bound)
        relaxed_solution = self.relaxed_solver.solve(target)
        preds = coating_to_reflective_props(relaxed_solution)
        error = match(preds, target)
        return TreeNode(relaxed_solution, error, lower_bound, upper_bound)

    def select_branching_index(self, refractive_indices: torch.Tensor):
        nondiscrete_indices = self.get_nondiscrete_indices(refractive_indices)
        return random.choice(nondiscrete_indices)
