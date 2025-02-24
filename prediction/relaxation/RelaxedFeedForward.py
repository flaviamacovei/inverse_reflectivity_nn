import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import wandb
import sys
sys.path.append(sys.path[0] + '/..')
from prediction.relaxation.BaseTrainableRelaxedSolver import BaseTrainableRelaxedSolver
from data.dataloaders.BaseDataloader import BaseDataloader
from data.values.Coating import Coating
from forward.forward_tmm import coating_to_reflective_props
from evaluation.loss import compute_loss
from data.values.ReflectivePropsPattern import ReflectivePropsPattern
from config import wavelengths, device, learning_rate, num_epochs

class TrainableMLP(nn.Module):
    def __init__(self, num_layers: int):
        super().__init__()
        in_dim = 2 * wavelengths.size()[0]
        layer_1_features = 128
        layer_2_features = 512
        layer_3_features = 64
        output_size = 2 * num_layers
        self.net = nn.Sequential(
            nn.Linear(in_dim, layer_1_features, device = device),
            nn.ReLU(),
            nn.Linear(layer_1_features, layer_2_features, device = device),
            nn.ReLU(),
            nn.Linear(layer_2_features, layer_3_features, device = device),
            nn.ReLU(),
            nn.Linear(layer_3_features, output_size, device = device)
        )

    def forward(self, x):
        return self.net(x)

class BoundedMLP(nn.Module):
    def __init__(self, trainable_model: TrainableMLP, thicknesses_bounds: (torch.Tensor, torch.Tensor), refractive_indices_bounds: (torch.Tensor, torch.Tensor)):
        super().__init__()
        self.trainable_model = trainable_model
        self.lower_bound = torch.cat((thicknesses_bounds[0], refractive_indices_bounds[0]), dim = 1)
        self.upper_bound = torch.cat((thicknesses_bounds[1], refractive_indices_bounds[1]), dim = 1)

    def forward(self, x):
        pretrained_output = self.trainable_model(x)
        scaled_output = (self.upper_bound - self.lower_bound) * torch.sigmoid(pretrained_output) + self.lower_bound
        return scaled_output

    def set_thicknesses_bounds(self, lower_bound: torch.Tensor, upper_bound: torch.Tensor):
        self.lower_bound[:, :wavelengths.size()[0]] = lower_bound
        self.upper_bound[:, :wavelengths.size()[0]] = upper_bound

    def set_refractive_indices_bounds(self, lower_bound: torch.Tensor, upper_bound: torch.Tensor):
        self.lower_bound[:, wavelengths.size()[0]:] = lower_bound
        self.upper_bound[:, wavelengths.size()[0]:] = upper_bound

class RelaxedFeedForward(BaseTrainableRelaxedSolver):
    def __init__(self, dataloader: BaseDataloader, num_layers: int, thicknesses_bounds: (torch.Tensor, torch.Tensor), refractive_indices_bounds: (torch.Tensor, torch.Tensor)):
        super().__init__(dataloader)
        self.SCALING_FACTOR_THICKNESSES = 1.0e4
        self.SCALING_FACTOR_REFRACTIVE_INDICES = 0.1
        self.trainable_model = TrainableMLP(num_layers).to(device)
        thicknesses_bounds = [bound * self.SCALING_FACTOR_THICKNESSES for bound in thicknesses_bounds]
        refractive_indices_bounds = [bound * self.SCALING_FACTOR_REFRACTIVE_INDICES for bound in refractive_indices_bounds]
        self.bounded_model = BoundedMLP(self.trainable_model, thicknesses_bounds, refractive_indices_bounds).to(device)
        self.optimiser = optim.Adam(self.bounded_model.parameters(), lr = learning_rate)

    def train(self):
        self.bounded_model.train()
        self.trainable_model.train()
        loss_scale = 1
        for epoch in range(num_epochs):
            self.optimiser.zero_grad()
            loss = torch.tensor(0.0, device = device)
            for refs in self.dataloader:
                refs = refs[0].float().to(device)
                coating = self.model_forward(refs)
                preds = coating_to_reflective_props(coating)
                lower_bound, upper_bound = refs.chunk(2, dim = 1)

                refs_obj = ReflectivePropsPattern(lower_bound, upper_bound)
                batch_loss = compute_loss(preds, refs_obj)
                loss += batch_loss

            if epoch == 0:
                loss_scale = loss
            # wandb.log({"loss": loss.item() / loss_scale})

            loss.backward()
            self.trainable_model.net[2].weight.grad[:, :wavelengths.size()[0]] /= self.SCALING_FACTOR_THICKNESSES
            self.trainable_model.net[2].weight.grad[:, wavelengths.size()[0]:] /= self.SCALING_FACTOR_REFRACTIVE_INDICES
            if epoch % 10 == 0:
                print(f"Loss in epoch {epoch + 1}: {loss.item()}")
            self.optimiser.step()

    def solve(self, target: ReflectivePropsPattern):
        self.bounded_model.eval()
        self.trainable_model.eval()
        model_input = torch.cat((target.get_lower_bound(), target.get_upper_bound()), dim = 1)
        return self.model_forward(model_input)

    def model_forward(self, target: torch.Tensor):
        coating_props = self.bounded_model(target)
        thicknesses, refractive_indices = coating_props.chunk(2, dim = 1)
        thicknesses = thicknesses / self.SCALING_FACTOR_THICKNESSES
        refractive_indices = refractive_indices / self.SCALING_FACTOR_REFRACTIVE_INDICES
        return Coating(thicknesses, refractive_indices)

    def set_bounds(self, thicknesses_bounds: (torch.Tensor, torch.Tensor), refractive_indices_bounds: (torch.Tensor, torch.Tensor)):
        thicknesses_bounds = [bound * self.SCALING_FACTOR_THICKNESSES for bound in thicknesses_bounds]
        refractive_indices_bounds = [bound * self.SCALING_FACTOR_REFRACTIVE_INDICES for bound in refractive_indices_bounds]
        self.bounded_model.set_thicknesses_bounds(thicknesses_bounds[0], thicknesses_bounds[1])
        self.bounded_model.set_refractive_indices_bounds(refractive_indices_bounds[0], refractive_indices_bounds[1])
