import torch
import torch.nn as nn
import sys
sys.path.append(sys.path[0] + '/..')
from prediction.BaseTrainableModel import BaseTrainableModel
from data.dataloaders.BaseDataloader import BaseDataloader
from utils.ConfigManager import ConfigManager as CM

class TrainableMLP(nn.Module):
    def __init__(self):
        super().__init__()
        in_dim = 2 * CM().get('wavelengths').size()[0]
        layer_1_features = 128
        layer_2_features = 512
        layer_3_features = 64
        self.output_size = (CM().get('layers.max') + 2) * (CM().get('material_embedding.dim') + 1)
        self.net = nn.Sequential(
            nn.Linear(in_dim, layer_1_features, device = CM().get('device')),
            nn.ReLU(),
            nn.BatchNorm1d(layer_1_features),
            nn.Linear(layer_1_features, layer_2_features, device = CM().get('device')),
            nn.ReLU(),
            nn.BatchNorm1d(layer_2_features),
            nn.Linear(layer_2_features, layer_3_features, device = CM().get('device')),
            nn.ReLU(),
            nn.BatchNorm1d(layer_3_features),
            nn.Linear(layer_3_features, self.output_size, device = CM().get('device')),
            nn.ReLU()
        )

    def forward(self, x):
        return torch.abs(self.net(x))

    def get_output_size(self):
        return self.output_size

class MLP(BaseTrainableModel):
    def __init__(self, dataloader: BaseDataloader = None):
        super().__init__(TrainableMLP().to(CM().get('device')), dataloader)

    def scale_gradients(self):
        pass
