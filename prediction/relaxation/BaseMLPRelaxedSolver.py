from torch import nn
import torch
import sys
sys.path.append(sys.path[0] + '/..')
from prediction.relaxation.BaseTrainableRelaxedSolver import BaseTrainableRelaxedSolver
from data.dataloaders.BaseDataloader import BaseDataloader
from utils.ConfigManager import ConfigManager as CM

class BoundedMLP(nn.Module):
    def __init__(self, trainable_model: nn.Module, output_size: int):
        super().__init__()
        self.output_size = output_size

        self.trainable_model = trainable_model
        pretreated_size = self.trainable_model.get_output_size()
        self.linear = nn.Linear(pretreated_size, self.output_size, device = CM().get('device'))

        self.lower_bound = torch.zeros((1, self.output_size), device = CM().get('device'))
        self.upper_bound = torch.ones((1, self.output_size), device = CM().get('device'))

    def forward(self, x):
        pretreated_result = self.trainable_model(x)
        scaled_result = self.linear(pretreated_result)
        norm_result = self.split_minmax_normalisation(scaled_result)
        output = (self.upper_bound - self.lower_bound) * norm_result + self.lower_bound
        return output

    def get_lower_bound(self):
        return self.lower_bound

    def get_upper_bound(self):
        return self.upper_bound

    def get_output_size(self):
        return self.output_size

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

        self.trainable_model = None
        self.model = None

    def initialise_model(self):
        self.model = BoundedMLP(self.trainable_model, 2 * self.num_layers).to(CM().get('device'))
        thicknesses_lower_bound = torch.ones((1, self.num_layers), device=CM().get('device')) * CM().get(
            'thicknesses_bounds.min')
        thicknesses_upper_bound = torch.ones((1, self.num_layers), device=CM().get('device')) * CM().get(
            'thicknesses_bounds.max')
        refractive_index_lower_bound = torch.ones((1, self.num_layers), device=CM().get('device')) * CM().get(
            'refractive_indices_bounds.min')
        refractive_index_upper_bound = torch.ones((1, self.num_layers), device=CM().get('device')) * CM().get(
            'refractive_indices_bounds.max')
        lower_bound = torch.cat((thicknesses_lower_bound * self.SCALING_FACTOR_THICKNESSES,
                                 refractive_index_lower_bound * self.SCALING_FACTOR_REFRACTIVE_INDICES), dim=1)
        upper_bound = torch.cat((thicknesses_upper_bound * self.SCALING_FACTOR_THICKNESSES,
                                 refractive_index_upper_bound * self.SCALING_FACTOR_REFRACTIVE_INDICES), dim=1)

        self.model.set_lower_bound(lower_bound)
        self.model.set_upper_bound(upper_bound)

    def set_lower_bound(self, lower_bound: torch.Tensor):
        thicknesses_lower_bound = lower_bound[:, :self.num_layers] * self.SCALING_FACTOR_THICKNESSES
        refractive_indices_lower_bound = lower_bound[:, self.num_layers:] * self.SCALING_FACTOR_REFRACTIVE_INDICES
        self.model.set_lower_bound(torch.cat((thicknesses_lower_bound, refractive_indices_lower_bound), dim = 1))

    def set_upper_bound(self, upper_bound: torch.Tensor):
        thicknesses_upper_bound = upper_bound[:, :self.num_layers] * self.SCALING_FACTOR_THICKNESSES
        refractive_indices_upper_bound = upper_bound[:, self.num_layers:] * self.SCALING_FACTOR_REFRACTIVE_INDICES
        self.model.set_upper_bound(torch.cat((thicknesses_upper_bound, refractive_indices_upper_bound), dim = 1))

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
