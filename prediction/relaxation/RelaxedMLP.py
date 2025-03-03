import torch
import torch.nn as nn
import sys
sys.path.append(sys.path[0] + '/..')
from prediction.relaxation.BaseMLPRelaxedSolver import BaseMLPRelaxedSolver
from data.dataloaders.BaseDataloader import BaseDataloader
from utils.ConfigManager import ConfigManager as CM

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

class RelaxedMLP(BaseMLPRelaxedSolver):
    def __init__(self, dataloader: BaseDataloader, num_layers: int):
        super().__init__(dataloader, num_layers)
        self.trainable_model = TrainableMLP().to(CM().get('device'))
        self.initialise_model()
        self.initialise_opitimiser()

    def scale_gradients(self):
        self.trainable_model.net[2].weight.grad[:,
        :CM().get('wavelengths').size()[0]] /= self.SCALING_FACTOR_THICKNESSES
        self.trainable_model.net[2].weight.grad[:,
        CM().get('wavelengths').size()[0]:] /= self.SCALING_FACTOR_REFRACTIVE_INDICES

