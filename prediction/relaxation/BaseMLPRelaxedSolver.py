from torch import nn
import torch
import sys
sys.path.append(sys.path[0] + '/..')
from prediction.relaxation.BaseTrainableRelaxedSolver import BaseTrainableRelaxedSolver
from data.dataloaders.BaseDataloader import BaseDataloader
from utils.ConfigManager import ConfigManager as CM

class BoundedMLP(nn.Module):
    def __init__(self, trainable_model: nn.Module):
        super().__init__()

        self.trainable_model = trainable_model
        self.size = self.trainable_model.get_output_size()
        self.linear = nn.Linear(self.size, self.size, device = CM().get('device'))

        self.lower_bound = torch.zeros((CM().get('training.batch_size'), self.size), device = CM().get('device'))
        self.upper_bound = torch.ones((CM().get('training.batch_size'), self.size), device = CM().get('device'))

    def forward(self, x):
        pretreated_result = self.trainable_model(x)
        scaled_result = self.linear(pretreated_result)
        norm_result = self.split_minmax_normalisation(scaled_result)
        # TODO: make this prettier (adjust to batch size)
        lower_bound = self.lower_bound[:x.shape[0]]
        upper_bound = self.upper_bound[:x.shape[0]]
        output = (upper_bound - lower_bound) * norm_result + lower_bound
        return output

    def get_lower_bound(self):
        return self.lower_bound

    def get_upper_bound(self):
        return self.upper_bound

    def get_size(self):
        return self.size

    def set_lower_bound(self, lower_bound: torch.Tensor):
        self.lower_bound = lower_bound

    def set_upper_bound(self, upper_bound: torch.Tensor):
        self.upper_bound = upper_bound

    def split_minmax_normalisation(self, x: torch.Tensor, epsilon = 1e-8):
        first_half, second_half = torch.chunk(x, 2, dim = 1)

        def minmax_norm(t: torch.Tensor):
            min_t = t.min(dim = -1, keepdim = True).values
            max_t = t.max(dim = -1, keepdim = True).values
            return (t - min_t) / (max_t - min_t + epsilon)  # Normalization

        first_half_norm = minmax_norm(first_half)
        second_half_norm = minmax_norm(second_half)

        return torch.cat([first_half_norm, second_half_norm], dim=-1)

class BaseMLPRelaxedSolver(BaseTrainableRelaxedSolver):
    def __init__(self, dataloader: BaseDataloader):
        super().__init__(dataloader)
        self.num_layers = CM().get('num_layers')
        self.batch_size = CM().get('training.batch_size')

        self.trainable_model = None
        self.model = None

    def initialise_model(self):
        self.model = BoundedMLP(self.trainable_model).to(CM().get('device'))
        # TODO: clean up
        thicknesses_lower_bound = torch.ones((self.batch_size, self.num_layers, 1), device=CM().get('device')) * CM().get(
            'thicknesses_bounds.min')
        thicknesses_upper_bound = torch.ones((self.batch_size, self.num_layers, 1), device=CM().get('device')) * CM().get(
            'thicknesses_bounds.max')
        refractive_index_lower_bound = torch.ones((self.batch_size, self.num_layers, CM().get('material_embedding.dim')), device=CM().get('device')) * CM().get(
            'refractive_indices_bounds.min')
        refractive_index_upper_bound = torch.ones((self.batch_size, self.num_layers, CM().get('material_embedding.dim')), device=CM().get('device')) * CM().get(
            'refractive_indices_bounds.max')
        lower_bound = torch.cat((thicknesses_lower_bound * self.SCALING_FACTOR_THICKNESSES,
                                 refractive_index_lower_bound * self.SCALING_FACTOR_REFRACTIVE_INDICES), dim=2)
        lower_bound = lower_bound.reshape((lower_bound.shape[0], lower_bound.shape[1] * lower_bound.shape[2]))
        upper_bound = torch.cat((thicknesses_upper_bound * self.SCALING_FACTOR_THICKNESSES,
                                 refractive_index_upper_bound * self.SCALING_FACTOR_REFRACTIVE_INDICES), dim=2)
        upper_bound = upper_bound.reshape((upper_bound.shape[0], upper_bound.shape[1] * upper_bound.shape[2]))

        self.model.set_lower_bound(lower_bound)
        self.model.set_upper_bound(upper_bound)

    def set_lower_bound(self, lower_bound: torch.Tensor):
        # TODO: scale
        # thicknesses_lower_bound = lower_bound[:, :self.num_layers] * self.SCALING_FACTOR_THICKNESSES
        # refractive_indices_lower_bound = lower_bound[:, self.num_layers:] * self.SCALING_FACTOR_REFRACTIVE_INDICES
        # self.model.set_lower_bound(torch.cat((thicknesses_lower_bound, refractive_indices_lower_bound), dim = 1))
        self.model.set_lower_bound(lower_bound)

    def set_upper_bound(self, upper_bound: torch.Tensor):
        # TODO: scale
        # thicknesses_upper_bound = upper_bound[:, :self.num_layers] * self.SCALING_FACTOR_THICKNESSES
        # refractive_indices_upper_bound = upper_bound[:, self.num_layers:] * self.SCALING_FACTOR_REFRACTIVE_INDICES
        # self.model.set_upper_bound(torch.cat((thicknesses_upper_bound, refractive_indices_upper_bound), dim = 1))
        self.model.set_upper_bound(upper_bound)

    def get_lower_bound(self):
        thicknesses_lower_bound = self.model.get_lower_bound()[:, :self.num_layers] / self.SCALING_FACTOR_THICKNESSES
        refractive_indices_lower_bound = self.model.get_lower_bound()[:, self.num_layers:] / self.SCALING_FACTOR_REFRACTIVE_INDICES
        return torch.cat((thicknesses_lower_bound, refractive_indices_lower_bound), dim = 1)

    def get_upper_bound(self):
        thicknesses_upper_bound = self.model.get_upper_bound()[:, :self.num_layers] / self.SCALING_FACTOR_THICKNESSES
        refractive_indices_upper_bound = self.model.get_upper_bound()[:, self.num_layers:] / self.SCALING_FACTOR_REFRACTIVE_INDICES
        return torch.cat((thicknesses_upper_bound, refractive_indices_upper_bound), dim = 1)

    def set_to_train(self):
        self.trainable_model.train()

    def set_to_eval(self):
        self.trainable_model.eval()
