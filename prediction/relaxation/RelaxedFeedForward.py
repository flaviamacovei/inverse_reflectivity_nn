import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import wandb
import gc
import sys
sys.path.append(sys.path[0] + '/..')
from prediction.relaxation.BaseTrainableRelaxedSolver import BaseTrainableRelaxedSolver
from data.dataloaders.BaseDataloader import BaseDataloader
from data.values.Coating import Coating
from forward.forward_tmm import coating_to_reflective_props
from evaluation.loss import match
from data.values.ReflectivePropsPattern import ReflectivePropsPattern
from utils.ConfigManager import ConfigManager as CM
from ui.visualise import visualise
from utils.os_utils import get_unique_filename

class TrainableMLP(nn.Module):
    def __init__(self):
        super().__init__()
        in_dim = 2 * CM().get('wavelengths').size()[0]
        layer_1_features = 128
        layer_2_features = 512
        layer_3_features = 64
        self.output_size = 20
        self.net = nn.Sequential(
            nn.Linear(in_dim, layer_1_features, device = CM().get('device')),
            nn.ReLU(),
            nn.Linear(layer_1_features, layer_2_features, device = CM().get('device')),
            nn.ReLU(),
            nn.Linear(layer_2_features, layer_3_features, device = CM().get('device')),
            nn.ReLU(),
            nn.Linear(layer_3_features, self.output_size, device = CM().get('device'))
        )

    def forward(self, x):
        return self.net(x)

    def get_output_size(self):
        return self.output_size

class BoundedMLP(nn.Module):
    def __init__(self, trainable_model: TrainableMLP, output_size: int):
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

class RelaxedFeedForward(BaseTrainableRelaxedSolver):
    def __init__(self, dataloader: BaseDataloader, num_layers: int):
        super().__init__(dataloader)
        self.trainable_model = TrainableMLP().to(CM().get('device'))
        self.model = BoundedMLP(self.trainable_model, 2 * num_layers).to(CM().get('device'))

        self.num_layers = num_layers

        thicknesses_lower_bound = torch.ones((1, num_layers), device=CM().get('device')) * CM().get(
            'thicknesses_bounds.min')
        thicknesses_upper_bound = torch.ones((1, num_layers), device=CM().get('device')) * CM().get(
            'thicknesses_bounds.max')
        refractive_index_lower_bound = torch.ones((1, num_layers), device=CM().get('device')) * CM().get(
            'refractive_indices_bounds.min')
        refractive_index_upper_bound = torch.ones((1, num_layers), device=CM().get('device')) * CM().get(
            'refractive_indices_bounds.max')
        lower_bound = torch.cat((thicknesses_lower_bound * self.SCALING_FACTOR_THICKNESSES,
                                 refractive_index_lower_bound * self.SCALING_FACTOR_REFRACTIVE_INDICES), dim=1)
        upper_bound = torch.cat((thicknesses_upper_bound * self.SCALING_FACTOR_THICKNESSES,
                                 refractive_index_upper_bound * self.SCALING_FACTOR_REFRACTIVE_INDICES), dim=1)

        self.model.set_lower_bound(lower_bound)
        self.model.set_upper_bound(upper_bound)

        self.initialise_opitimiser()

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

    def scale_gradients(self):
        self.trainable_model.net[2].weight.grad[:,
        :CM().get('wavelengths').size()[0]] /= self.SCALING_FACTOR_THICKNESSES
        self.trainable_model.net[2].weight.grad[:,
        CM().get('wavelengths').size()[0]:] /= self.SCALING_FACTOR_REFRACTIVE_INDICES

