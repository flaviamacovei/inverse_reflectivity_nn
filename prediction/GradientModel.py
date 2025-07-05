import torch
import torch.optim as optim
import numpy as np
from scipy.optimize import minimize, Bounds
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


    def predict(self, target: ReflectivePropsPattern):
        """
        Predict a coating given a reflective properties pattern object.

        Args:
            target: Reflective properties pattern for which to perform prediction.
        """
        init_params = self.initialise()
        bounds = Bounds(np.zeros_like(init_params), np.ones_like(init_params))

        def np_loss_function(params: np.ndarray):
            params = torch.from_numpy(params)
            loss, grads = self.loss_function(params, target)
            return loss.cpu().detach().numpy(), grads.cpu().detach().numpy()

        result = minimize(
            fun = np_loss_function,
            x0 = init_params,
            method = 'L-BFGS-B',
            jac = True,
            bounds = bounds
        )
        optimised_params = result.x

        optimised_params = torch.from_numpy(optimised_params).reshape(self.coating_length, self.encoding_length)[None, :, :]
        optimised_params = optimised_params.to(CM().get('device')).float()

        return Coating(optimised_params)


    def initialise(self):
        init_params = np.random.randn(self.coating_length, self.encoding_length).flatten()
        return init_params


    def loss_function(self, params: torch.Tensor, target: ReflectivePropsPattern):
        params = params.reshape(self.coating_length, self.encoding_length)[None, :, :]
        params = params.to(CM().get('device')).float()
        params.requires_grad_()
        coating = Coating(params)
        preds = coating_to_reflective_props(coating)

        loss = match(preds, target)
        loss.backward()

        grads = params.grad.flatten()
        print(f"loss:{loss.item()}")


        return loss, grads