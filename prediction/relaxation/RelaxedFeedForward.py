import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append(sys.path[0] + '/..')
from prediction.relaxation.BaseTrainableRelaxedSolver import BaseTrainableRelaxedSolver
from data.dataloaders.BaseDataloader import BaseDataloader
from data.values.Coating import Coating
from forward.forward_tmm import coating_to_reflective_props
from evaluation.loss import compute_loss
from data.values.ReflectivePropsPattern import ReflectivePropsPattern
from config import wavelengths, device, learning_rate, num_epochs

class FeedForwardNet(nn.Module):
    def __init__(self, num_layers: int):
        super().__init__()
        in_dim = 2 * wavelengths.size()[0]
        layer_1_features = 100
        layer_2_features = 20
        output_size = 2 * num_layers
        self.net = nn.Sequential(
            nn.Linear(in_dim, layer_1_features, device = device),
            nn.ReLU(),
            nn.Linear(layer_1_features, layer_2_features, device = device),
            nn.ReLU(),
            nn.Linear(layer_2_features, output_size, device = device)
        )

    def forward(self, x):
        return self.net(x).squeeze()

class RelaxedFeedForward(BaseTrainableRelaxedSolver):
    def __init__(self, dataloader: BaseDataloader, num_layers: int):
        super().__init__(dataloader)
        self.model = FeedForwardNet(num_layers).to(device)
        self.optimiser = optim.Adam(self.model.parameters(), lr = learning_rate)
        self.SCALING_FACTOR_THICKNESSES = 1.0e6
        self.SCALING_FACTOR_REFRACTIVE_INDICES = 10

    def train(self):
        self.model.train()
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

            loss.backward()
            self.model.net[2].weight.grad[:, :wavelengths.size()[0]] /= self.SCALING_FACTOR_THICKNESSES
            self.model.net[2].weight.grad[:, wavelengths.size()[0]:] *= self.SCALING_FACTOR_REFRACTIVE_INDICES
            if epoch % 10 == 0:
                print(f"Loss in epoch {epoch + 1}: {loss.item()}")
            self.optimiser.step()

    def solve_relaxed(self, target: ReflectivePropsPattern):
        self.model.eval()
        model_input = torch.cat((target.get_lower_bound(), target.get_upper_bound()), dim = 0)
        return self.model_forward(model_input)

    def model_forward(self, target: torch.Tensor):
        coating_props = self.model(target)
        thicknesses, refractive_indices = coating_props.chunk(2, dim = 0)
        thicknesses = thicknesses / self.SCALING_FACTOR_THICKNESSES
        refractive_indices = refractive_indices * self.SCALING_FACTOR_REFRACTIVE_INDICES
        return Coating(thicknesses, refractive_indices)
