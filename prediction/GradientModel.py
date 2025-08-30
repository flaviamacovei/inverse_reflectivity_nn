import torch
import numpy as np
import time
from scipy.optimize import minimize, Bounds, check_grad
import sys
sys.path.append(sys.path[0] + '/..')
from prediction.BaseModel import BaseModel
from data.values.ReflectivityPattern import ReflectivityPattern
from utils.ConfigManager import ConfigManager as CM
from data.values.Coating import Coating
from forward.forward_tmm import coating_to_reflectivity
from evaluation.loss import match

class GradientModel(BaseModel):
    def __init__(self):
        super().__init__()
        # number of thin films plus substrate and air
        self.coating_length = CM().get('num_layers') + 2
        # thicknesses plus embedding dimension
        self.encoding_length = CM().get('material_embedding.dim') + 1


    def predict(self, target: ReflectivityPattern):
        """
        Predict a coating given a reflectivity pattern object.

        Args:
            target: Reflectivity pattern for which to perform prediction.
        """
        batch_size = target.get_batch_size()
        self.initialise(batch_size = batch_size)
        bounds = Bounds(np.zeros_like(self.init_params), np.ones_like(self.init_params))

        def np_loss_function(params: np.ndarray):
            params = torch.tensor(params, dtype=torch.float32, requires_grad=True)
            loss, grads = self.loss_function(params, target)
            return loss.detach().cpu().numpy(), grads.detach().cpu().numpy()

        start_time = time.time()
        result = minimize(
            fun = np_loss_function,
            x0 = self.init_params,
            method = 'L-BFGS-B',
            jac = True,
            bounds = bounds,
            options = {"maxiter": 1000}
        )
        end_time = time.time()
        print(f"this shit took {end_time - start_time} seconds")

        optimised_params = result.x

        optimised_params = torch.from_numpy(optimised_params).reshape(batch_size, self.coating_length, self.encoding_length).float()
        optimised_params = optimised_params.to(CM().get('device'))

        return Coating(optimised_params)

    def initialise(self, init_params = None, batch_size: int = None):
        assert init_params is not None or batch_size is not None
        if init_params is None:
            init_params = np.random.randn(batch_size, self.coating_length, self.encoding_length).flatten()
        self.init_params = init_params


    def loss_function(self, params: torch.Tensor, target: ReflectivityPattern):
        flat_params = params.detach().clone().requires_grad_(True)
        original = flat_params
        if flat_params.shape[0] == self.coating_length * self.encoding_length:
            params = flat_params.reshape(self.coating_length, self.encoding_length)[None]
        elif flat_params.shape[0] == len(target) * self.coating_length * self.encoding_length:
            params = flat_params.reshape(len(target), self.coating_length, self.encoding_length)
        params = params.to(CM().get('device'))

        coating = Coating(params)
        preds = coating_to_reflectivity(coating)
        loss = match(preds, target)
        loss.backward()

        grads = original.grad.flatten()

        # print(f"loss: {loss}, grad norm: {torch.linalg.norm(grads):.3e}")
        return loss, grads
