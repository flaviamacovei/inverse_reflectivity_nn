import math
import torch
import torch.nn as nn
import sys
sys.path.append(sys.path[0] + '/..')
from prediction.BaseTrainableModel import BaseTrainableModel, ThicknessPostProcess
from data.dataloaders.BaseDataloader import BaseDataloader
from utils.ConfigManager import ConfigManager as CM

class TrainableMLP(nn.Module):
    """
    Trainable multilayer perceptron. Extends nn.Module.

    Architecture:
        2 x |wavelengths| --> ...hidden dimensions... --> out_dim
    See readme for details.

    Attributes:
        out_dim: Output size of the network. Equal to |coating| * (embedding_dim + 1)

    Methods:
        forward: Propagate input through the model.
        get_output_size: Return output size of the network.
    """
    def __init__(self, in_dims: dict, out_dims: dict):
        """Initialise a TrainableMLP instance."""
        super().__init__()
        self.in_dim = in_dims['seq_len'] * in_dims['dim']
        self.out_dims = out_dims
        dimensions = [self.in_dim] + CM().get('mlp.hidden_dims')

        shared_layers = []
        for i in range(len(dimensions) - 1):
            shared_layers.append(nn.Linear(dimensions[i], dimensions[i + 1]))
            shared_layers.append(nn.ReLU())
            shared_layers.append(nn.BatchNorm1d(dimensions[i + 1]))
        self.shared_net = nn.ModuleList(shared_layers)
        self.thickness_head = nn.Sequential(
            nn.Linear(dimensions[-1], out_dims['seq_len'] * out_dims['thickness']),
            ThicknessPostProcess(out_dims['seq_len'])
        )

        self.material_head = nn.Sequential(
            nn.Linear(dimensions[-1], out_dims['seq_len'] * out_dims['material']),
            nn.ReLU(),
        )

    def forward(self, x):
        """Propagate input through the model."""
        for layer in self.shared_net:
            x = layer(x)
        thickness_outputs = self.thickness_head(x)
        material_outputs = self.material_head(x)
        return thickness_outputs, material_outputs

    def get_output_size(self):
        """Return output size of the network."""
        return self.out_dims

class MLP(BaseTrainableModel):
    """
    Trainable prediction model using an MLP as base.

    Attributes:
        model: Instance of TrainableMLP.
    """
    def __init__(self):
        super().__init__()
        """Initialise an MLP instance."""

    def build_model(self):

        return TrainableMLP(self.in_dims, self.out_dims).to(CM().get('device'))

    def get_model_output(self, src, tgt = None):
        """
        Get output of the model for given input.

        For the MLP architecture, only src is needed. tgt is leftover from superclass signature and not used.

        Args:
            src: Input data.
            tgt: Target data. Ignore.

        Returns:
            Output of the model.
        """
        src = src.reshape(-1, self.in_dims['seq_len'] * self.in_dims['dim'])
        out_thicknesses, out_materials = self.model(src)
        out_thicknesses = out_thicknesses.reshape(-1, self.tgt_seq_len, 1)
        out_materials = out_materials.reshape(-1, self.tgt_seq_len, self.tgt_vocab_size)
        return out_thicknesses, out_materials

    def get_architecture_name(self):
        """
        Return name of model architecture.
        """
        return "mlp"

    def get_shared_params(self):
        params = []
        for param in self.model.shared_net.parameters():
            params.append(param)
        return params

    def get_thicknesses_params(self):
        params = []
        for param in self.model.thickness_head.parameters():
            params.append(param)
        return params

    def get_materials_params(self):
        params = []
        for param in self.model.material_head.parameters():
            params.append(param)
        return params