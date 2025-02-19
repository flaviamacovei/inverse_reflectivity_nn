import torch
import numpy as np
from scipy.optimize import minimize, Bounds
import sys
sys.path.append(sys.path[0] + '/..')
from prediction.relaxation.BaseRelaxedSolver import BaseRelaxedSolver
from data.values.ReflectivePropsPattern import ReflectivePropsPattern
from data.values.Coating import Coating
from forward.forward_tmm import coating_to_reflective_props
from evaluation.loss import compute_loss
from config import device

class GradientSolver(BaseRelaxedSolver):
    def __init__(self, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.SCALING_FACTOR = 1.0e9
        self.MIN_REFRACTIVE_INDICES = 1.0
        self.MAX_REFRACTIVE_INDICES = 3.5
        self.MIN_THICKNESSES = 10
        self.MAX_THICKNESSES = 10_000

    def solve_relaxed(self, target: ReflectivePropsPattern):
        params_init = self.make_init()
        bounds = self.make_bounds(self.num_layers)

        result = minimize(
            fun=self.loss_function,
            x0=params_init,
            args=(target),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds
        )

        optimised_params = result.x

        optimised_refractive_indices, optimised_thicknesses = self.split_params(optimised_params)
        optimised_thicknesses_scaled = optimised_thicknesses / self.SCALING_FACTOR
        optimal_coating = Coating(optimised_thicknesses_scaled, optimised_refractive_indices)

        return optimal_coating

    def loss_function(self, params, refs: ReflectivePropsPattern):
        refractive_indices, thicknesses = self.split_params(params)
        thicknesses_scaled = thicknesses / self.SCALING_FACTOR
        coating = Coating(thicknesses_scaled, refractive_indices)
        preds = coating_to_reflective_props(coating)
        loss = compute_loss(preds, refs)

        loss.backward(retain_graph = True)

        grad_refractive_indices = refractive_indices.grad.cpu().numpy().flatten()
        grad_thicknesses = (thicknesses.grad.cpu().numpy().flatten() / self.SCALING_FACTOR)

        return loss.item(), np.concatenate((grad_refractive_indices, grad_thicknesses))

    def make_init(self):
        thicknesses_init = 1 + torch.randn((self.num_layers))
        refractive_indices_init = 1 + torch.randn((self.num_layers))
        return np.concatenate((refractive_indices_init.flatten(), thicknesses_init.flatten()))

    def make_bounds(self, shape: int):
        bounds_refractive_indices_min = np.full(shape, self.MIN_REFRACTIVE_INDICES)
        bounds_refarctive_indices_max = np.full(shape, self.MAX_REFRACTIVE_INDICES)
        bounds_thicknesses_min = np.full(shape, self.MIN_THICKNESSES)
        bounds_thicknesses_max = np.full(shape, self.MAX_THICKNESSES)
        bounds_thicknesses_min[0] = self.SCALING_FACTOR
        bounds_thicknesses_max[0] = self.SCALING_FACTOR
        bounds_thicknesses_min[-1] = self.SCALING_FACTOR
        bounds_thicknesses_max[-1] = self.SCALING_FACTOR
        return Bounds(
            np.concatenate((bounds_refractive_indices_min, bounds_thicknesses_min)),
            np.concatenate((bounds_refarctive_indices_max, bounds_thicknesses_max))
        )

    def split_params(self, params):
        refractive_indices = torch.tensor(params[:np.prod(self.num_layers)].reshape(self.num_layers),
                                          dtype=torch.float64, device=device, requires_grad=True)
        thicknesses = torch.tensor(params[np.prod(self.num_layers):].reshape(self.num_layers), dtype=torch.float64,
                                   device=device, requires_grad=True)
        return refractive_indices, thicknesses
