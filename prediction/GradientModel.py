import torch
import numpy as np
from scipy.optimize import minimize, Bounds, check_grad
import sys
sys.path.append(sys.path[0] + '/..')
from prediction.BaseModel import BaseModel
from data.values.ReflectivePropsPattern import ReflectivePropsPattern
from utils.ConfigManager import ConfigManager as CM
from data.values.Coating import Coating
from forward.forward_tmm import coating_to_reflective_props
from evaluation.loss import match

class GradientModel(BaseModel):
    def __init__(self):
        super().__init__()
        # number of thin films plus substrate and air
        self.coating_length = CM().get('num_layers') + 2
        # thicknesses plus embedding dimension
        self.encoding_length = CM().get('material_embedding.dim') + 1
        self.initialise()


    def predict(self, target: ReflectivePropsPattern):
        """
        Predict a coating given a reflective properties pattern object.

        Args:
            target: Reflective properties pattern for which to perform prediction.
        """
        bounds = Bounds(np.zeros_like(self.init_params), np.ones_like(self.init_params))

        def np_loss_function(params: np.ndarray):
            params = torch.tensor(params, dtype=torch.float32, requires_grad=True)
            loss, grads = self.loss_function(params, target)
            return loss.detach().cpu().numpy(), grads.detach().cpu().numpy()

        result = minimize(
            fun = np_loss_function,
            x0 = self.init_params,
            method = 'L-BFGS-B',
            jac = True,
            bounds = bounds
        )

        optimised_params = result.x

        optimised_params = torch.from_numpy(optimised_params).reshape(self.coating_length, self.encoding_length).float()[None]
        optimised_params = optimised_params.to(CM().get('device'))

        return Coating(optimised_params)

    def initialise(self, init_params = None):
        if init_params == None:
            init_params = np.random.randn(self.coating_length, self.encoding_length).flatten()
        self.init_params = init_params


    def loss_function(self, params: torch.Tensor, target: ReflectivePropsPattern):
        flat_params = params.detach().clone().requires_grad_(True)
        original = flat_params
        if flat_params.shape[0] == self.coating_length * self.encoding_length:
            params = flat_params.reshape(self.coating_length, self.encoding_length)[None]
        elif flat_params.shape[0] == len(target) * self.coating_length * self.encoding_length:
            params = flat_params.reshape(len(target), self.coating_length, self.encoding_length)
        params = params.to(CM().get('device'))

        coating = Coating(params)
        preds = coating_to_reflective_props(coating)
        loss = match(preds, target)
        loss.backward()

        grads = original.grad.flatten()

        # print(f"loss: {loss}, grad norm: {torch.linalg.norm(grads):.3e}")
        return loss, grads
