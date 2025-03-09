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
from utils.ConfigManager import ConfigManager as CM

class BranchAndBound(BaseDiscretiser):
    def __init__(self, relaxed_solver: BaseRelaxedSolver):
        super().__init__(relaxed_solver)

    def predict(self, target: ReflectivePropsPattern):
        solution_tree = [self.make_node(target, self.relaxed_solver.get_lower_bound(), self.relaxed_solver.get_upper_bound())]
        value = coating_to_reflective_props(solution_tree[0].get_coating())
        # visualise(value, target, "before_rounding")
        max_error = float("Inf")
        optimum = None
        iteration = 0
        while len(solution_tree) > 0:
            current_node = solution_tree.pop()
            refractive_indices = current_node.get_coating().get_refractive_indices()
            if current_node.get_error() < max_error:
                if RefractiveIndex.is_discrete(refractive_indices):
                    optimum = current_node.get_coating()
                    max_error = current_node.get_error()
                else:
                    branching_indices = self.select_branching_indices(refractive_indices)
                    mask = torch.cat((torch.zeros_like(branching_indices), branching_indices), dim = 1).bool()

                    lower_bound = current_node.get_lower_bound()
                    upper_bound = current_node.get_upper_bound()

                    round_tensor = torch.cat((torch.zeros_like(refractive_indices), refractive_indices), dim = 1)

                    upper_bound_floor = upper_bound.clone()
                    upper_bound_floor[mask] = RefractiveIndex.floor_tensor(round_tensor)[mask]
                    floor_node = self.make_node(target, lower_bound, upper_bound_floor)
                    solution_tree.append(floor_node)

                    lower_bound_ceil = lower_bound.clone()
                    lower_bound_ceil[mask] = RefractiveIndex.ceil_tensor(round_tensor)[mask]
                    ceil_node = self.make_node(target, lower_bound_ceil, upper_bound)
                    solution_tree.append(ceil_node)

            iteration += 1
            if iteration == CM().get('branch_and_bound.max_iter'):
                if optimum:
                    # print("Maximum iterations reached, returning current optimum")
                    return optimum
                else:
                    # print("Maximum iterations reached, resorting to rounding")
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

    def select_branching_indices(self, refractive_indices: torch.Tensor):
        nondiscrete_map = self.get_nondiscrete_map(refractive_indices)
        rows = nondiscrete_map.nonzero(as_tuple=True)[0].unique()

        branching_indices = torch.zeros_like(refractive_indices)
        for row in rows:
            cols = (nondiscrete_map[row] == 1).nonzero(as_tuple=True)[0]
            if cols.numel() > 0:
                chosen_col = cols[torch.randint(len(cols), (1,))]
                branching_indices[row, chosen_col] = 1
        return branching_indices
