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
        self.SCALING_FACTOR_THICKNESSES = 1.0e6
        self.SCALING_FACTOR_REFRACTIVE_INDICES = 0.1
        self.num_layers = num_layers

        self.loss_functions = {
            "free": self.compute_loss_free,
            "guided": self.compute_loss_guided
        }
        self.compute_loss = self.loss_functions.get(CM().get('training.loss_function'))

        self.trainable_model = TrainableMLP().to(CM().get('device'))
        self.bounded_model = BoundedMLP(self.trainable_model, 2 * num_layers).to(CM().get('device'))

        thicknesses_lower_bound = torch.ones((1, num_layers), device=CM().get('device')) * CM().get('thicknesses_bounds.min')
        thicknesses_upper_bound = torch.ones((1, num_layers), device=CM().get('device')) * CM().get('thicknesses_bounds.max')
        refractive_index_lower_bound = torch.ones((1, num_layers), device=CM().get('device')) * CM().get('refractive_indices_bounds.min')
        refractive_index_upper_bound = torch.ones((1, num_layers), device=CM().get('device')) * CM().get('refractive_indices_bounds.max')
        lower_bound = torch.cat((thicknesses_lower_bound * self.SCALING_FACTOR_THICKNESSES, refractive_index_lower_bound * self.SCALING_FACTOR_REFRACTIVE_INDICES), dim = 1)
        upper_bound = torch.cat((thicknesses_upper_bound * self.SCALING_FACTOR_THICKNESSES, refractive_index_upper_bound * self.SCALING_FACTOR_REFRACTIVE_INDICES), dim = 1)

        self.bounded_model.set_lower_bound(lower_bound)
        self.bounded_model.set_upper_bound(upper_bound)

        self.optimiser = optim.Adam(self.bounded_model.parameters(), lr = CM().get('training.learning_rate'))

    def train(self):
        self.trainable_model.train()
        loss_scale = None
        for epoch in range(CM().get('training.num_epochs')):
            self.dataloader.next_epoch()
            self.optimiser.zero_grad()
            epoch_loss = torch.tensor(0.0, device = CM().get('device'))
            for batch in self.dataloader:
                torch.cuda.empty_cache()
                gc.collect()

                loss = self.compute_loss(batch)
                epoch_loss += loss

                if CM().get('wandb_log'):
                    if not loss_scale:
                        loss_scale = loss
                    wandb.log({"loss": loss.item() / loss_scale})

                loss.backward()
                self.trainable_model.net[2].weight.grad[:, :CM().get('wavelengths').size()[0]] /= self.SCALING_FACTOR_THICKNESSES
                self.trainable_model.net[2].weight.grad[:, CM().get('wavelengths').size()[0]:] /= self.SCALING_FACTOR_REFRACTIVE_INDICES

                self.optimiser.step()
            if epoch % 1 == 0:
                print(f"Loss in epoch {epoch + 1}: {epoch_loss.item()}")
        model_filename = f"out/models/model_{CM().get('training.loss_function')}_{'switch' if CM().get('training.dataset_switching') else 'no-switch'}_{CM().get('wavelengths').size()[0]}.pt"
        torch.save(self.bounded_model, get_unique_filename(model_filename))


    def solve(self, target: ReflectivePropsPattern):
        self.trainable_model.eval()
        model_input = torch.cat((target.get_lower_bound(), target.get_upper_bound()), dim = 1)
        return self.scaled_forward(model_input)

    def scaled_forward(self, target: torch.Tensor):
        coating_props = self.bounded_model(target)
        thicknesses, refractive_indices = coating_props.chunk(2, dim = 1)
        thicknesses = thicknesses / self.SCALING_FACTOR_THICKNESSES
        refractive_indices = refractive_indices / self.SCALING_FACTOR_REFRACTIVE_INDICES
        return Coating(thicknesses, refractive_indices)

    def compute_loss_guided(self, batch: (torch.Tensor, torch.Tensor)):
        pattern, labels = batch
        pattern = pattern.float().to(CM().get('device'))
        labels = labels.float().to(CM().get('device'))
        coating = self.scaled_forward(pattern)
        preds = torch.cat((coating.get_thicknesses(), coating.get_refractive_indices()), dim=1)
        return torch.sum((preds - labels)**2)**0.5

    def compute_loss_free(self, batch: torch.Tensor):
        pattern = batch[0].float().to(CM().get('device'))
        lower_bound, upper_bound = pattern.chunk(2, dim=1)
        refs_obj = ReflectivePropsPattern(lower_bound, upper_bound)
        coating = self.scaled_forward(pattern)
        preds = coating_to_reflective_props(coating)
        return match(preds, refs_obj)

    def set_lower_bound(self, lower_bound: torch.Tensor):
        thicknesses_lower_bound = lower_bound[:, :self.num_layers] * self.SCALING_FACTOR_THICKNESSES
        refractive_indices_lower_bound = lower_bound[:, self.num_layers:] * self.SCALING_FACTOR_REFRACTIVE_INDICES
        self.bounded_model.set_lower_bound(torch.cat((thicknesses_lower_bound, refractive_indices_lower_bound), dim = 1))

    def set_upper_bound(self, upper_bound: torch.Tensor):
        thicknesses_upper_bound = upper_bound[:, :self.num_layers] * self.SCALING_FACTOR_THICKNESSES
        refractive_indices_upper_bound = upper_bound[:, self.num_layers:] * self.SCALING_FACTOR_REFRACTIVE_INDICES
        self.bounded_model.set_upper_bound(torch.cat((thicknesses_upper_bound, refractive_indices_upper_bound), dim = 1))

    def get_lower_bound(self):
        thicknesses_lower_bound = self.bounded_model.get_lower_bound()[:, :self.num_layers] / self.SCALING_FACTOR_THICKNESSES
        refractive_indices_lower_bound = self.bounded_model.get_lower_bound()[:, self.num_layers:] / self.SCALING_FACTOR_REFRACTIVE_INDICES
        return torch.cat((thicknesses_lower_bound, refractive_indices_lower_bound), dim = 1)

    def get_upper_bound(self):
        thicknesses_upper_bound = self.bounded_model.get_upper_bound()[:, :self.num_layers] / self.SCALING_FACTOR_THICKNESSES
        refractive_indices_upper_bound = self.bounded_model.get_upper_bound()[:, self.num_layers:] / self.SCALING_FACTOR_REFRACTIVE_INDICES
        return torch.cat((thicknesses_upper_bound, refractive_indices_upper_bound), dim = 1)

