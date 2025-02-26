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
from evaluation.loss import compute_loss
from data.values.ReflectivePropsPattern import ReflectivePropsPattern
from config import wavelengths, device, learning_rate, num_epochs, thicknesses_bounds, refractive_indices_bounds

class TrainableMLP(nn.Module):
    def __init__(self):
        super().__init__()
        in_dim = 2 * wavelengths.size()[0]
        layer_1_features = 128
        layer_2_features = 512
        layer_3_features = 64
        self.output_size = 20
        self.net = nn.Sequential(
            nn.Linear(in_dim, layer_1_features, device = device),
            nn.ReLU(),
            nn.Linear(layer_1_features, layer_2_features, device = device),
            nn.ReLU(),
            nn.Linear(layer_2_features, layer_3_features, device = device),
            nn.ReLU(),
            nn.Linear(layer_3_features, self.output_size, device = device)
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
        self.linear = nn.Linear(pretreated_size, self.output_size, device = device)

        self.lower_bound = torch.zeros((1, self.output_size), device = device)
        self.upper_bound = torch.ones((1, self.output_size), device = device)

    def forward(self, x):
        pretreated_result = self.trainable_model(x)
        scaled_result = self.linear(pretreated_result)
        output = (self.upper_bound - self.lower_bound) * torch.sigmoid(scaled_result) + self.lower_bound
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

class RelaxedFeedForward(BaseTrainableRelaxedSolver):
    def __init__(self, dataloader: BaseDataloader, num_layers: int):
        super().__init__(dataloader)
        self.SCALING_FACTOR_THICKNESSES = 1.0e4
        self.SCALING_FACTOR_REFRACTIVE_INDICES = 0.1
        self.num_layers = num_layers

        self.trainable_model = TrainableMLP().to(device)
        self.bounded_model = BoundedMLP(self.trainable_model, 2 * num_layers).to(device)

        thicknesses_lower_bound = torch.ones((1, num_layers), device=device) * thicknesses_bounds[0]
        thicknesses_upper_bound = torch.ones((1, num_layers), device=device) * thicknesses_bounds[1]
        refractive_index_lower_bound = torch.ones((1, num_layers), device=device) * refractive_indices_bounds[0]
        refractive_index_upper_bound = torch.ones((1, num_layers), device=device) * refractive_indices_bounds[1]
        lower_bound = torch.cat((thicknesses_lower_bound * self.SCALING_FACTOR_THICKNESSES, refractive_index_lower_bound * self.SCALING_FACTOR_REFRACTIVE_INDICES), dim = 1)
        upper_bound = torch.cat((thicknesses_upper_bound * self.SCALING_FACTOR_THICKNESSES, refractive_index_upper_bound * self.SCALING_FACTOR_REFRACTIVE_INDICES), dim = 1)

        self.bounded_model.set_lower_bound(lower_bound)
        self.bounded_model.set_upper_bound(upper_bound)

        self.optimiser = optim.Adam(self.bounded_model.parameters(), lr = learning_rate)

    def train(self):
        self.trainable_model.train()
        loss_scale = 1
        for epoch in range(num_epochs):
            self.optimiser.zero_grad()
            epoch_loss = torch.tensor(0.0, device = device)
            for i, refs in enumerate(self.dataloader):
                # loss = torch.tensor(0.0, device = device)
                print(f"iteration: {i}")
                torch.cuda.empty_cache()
                gc.collect()
                # print(f"Before forward: {torch.cuda.memory_allocated() / 1e6} MB")

                refs = refs[0].float().to(device)
                coating = self.model_forward(refs)
                preds = coating_to_reflective_props(coating)
                lower_bound, upper_bound = refs.chunk(2, dim = 1)

                refs_obj = ReflectivePropsPattern(lower_bound, upper_bound)
                loss = compute_loss(preds, refs_obj)
                epoch_loss += loss
                # loss += batch_loss
                # print(f"should be None: {loss.grad_fn}")

                # print(f"After backward: {torch.cuda.memory_allocated() / 1e6} MB")

            # if epoch == 0:
            #     loss_scale = loss
            # wandb.log({"loss": loss.item() / loss_scale})

                loss.backward()
                self.trainable_model.net[2].weight.grad[:, :wavelengths.size()[0]] /= self.SCALING_FACTOR_THICKNESSES
                self.trainable_model.net[2].weight.grad[:, wavelengths.size()[0]:] /= self.SCALING_FACTOR_REFRACTIVE_INDICES
                self.optimiser.step()
            if epoch % 1 == 0:
                print(f"Loss in epoch {epoch + 1}: {epoch_loss.item()}")

    def solve(self, target: ReflectivePropsPattern):
        self.trainable_model.eval()
        model_input = torch.cat((target.get_lower_bound(), target.get_upper_bound()), dim = 1)
        return self.model_forward(model_input)

    def model_forward(self, target: torch.Tensor):
        coating_props = self.bounded_model(target)
        thicknesses, refractive_indices = coating_props.chunk(2, dim = 1)
        thicknesses = thicknesses / self.SCALING_FACTOR_THICKNESSES
        refractive_indices = refractive_indices / self.SCALING_FACTOR_REFRACTIVE_INDICES
        return Coating(thicknesses, refractive_indices)

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

