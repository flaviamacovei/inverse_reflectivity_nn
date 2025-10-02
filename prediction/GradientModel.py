import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import minimize, Bounds
import sys
sys.path.append(sys.path[0] + '/..')
from prediction.BaseModel import BaseModel
from data.values.ReflectivityPattern import ReflectivityPattern
from utils.ConfigManager import ConfigManager as CM
from data.values.Coating import Coating
from forward.forward_tmm import coating_to_reflectivity
from evaluation.loss import match
from data.material_embedding.EmbeddingManager import EmbeddingManager as EM

class GradientModel(BaseModel):
    def __init__(self):
        super().__init__()
        # thicknesses plus embedding dimension
        self.encoding_length = 2


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

        result = minimize(
            fun = np_loss_function,
            x0 = self.init_params,
            method = 'L-BFGS-B',
            jac = True,
            bounds = bounds,
            options = {"maxiter": 1000}
        )

        optimised_params = result.x

        optimised_params = torch.from_numpy(optimised_params).reshape(batch_size, self.out_dims['seq_len'], self.out_dims['thickness'] + self.out_dims['material']).float()
        optimised_params = optimised_params.to(CM().get('device'))

        return self.params_to_coating(optimised_params)

    def params_to_coating(self, params: torch.Tensor):
        thicknesses = params[:, :, :1]
        materials = params[:, :, 1:]
        softmax_probs = F.softmax(materials, dim = -1)
        _, material_indices = torch.max(softmax_probs, dim = -1, keepdim = True)
        coating = Coating(torch.cat([thicknesses, material_indices], dim = -1))
        return coating

    def initialise(self, init_params = None, batch_size: int = None):
        assert init_params is not None or batch_size is not None
        if init_params is None:
            init_params = np.random.randn(batch_size, self.out_dims['seq_len'], self.out_dims['thickness'] + self.out_dims['material']).flatten()
        self.init_params = init_params


    def loss_function(self, params: torch.Tensor, target: ReflectivityPattern):
        flat_params = params.detach().clone().requires_grad_(True)
        original = flat_params
        if flat_params.shape[0] == self.out_dims['seq_len'] * (self.out_dims['thickness'] + self.out_dims['material']):
            params = flat_params.reshape(self.out_dims['seq_len'], self.out_dims['thickness'] + self.out_dims['material'])[None]
        elif flat_params.shape[0] == len(target) * self.out_dims['seq_len'] * (self.out_dims['thickness'] + self.out_dims['material']):
            params = flat_params.reshape(len(target), self.out_dims['seq_len'], self.out_dims['thickness'] + self.out_dims['material'])
        params = params.to(CM().get('device'))

        coating = self.params_to_coating(params)
        preds = coating_to_reflectivity(coating)
        loss = match(preds, target)
        loss.backward()

        grads = original.grad.flatten()

        # print(f"loss: {loss}, grad norm: {torch.linalg.norm(grads):.3e}")
        return loss, grads

    def get_architecture_name(self):
        """
        Return name of model architecture.
        """
        return "gradient"

