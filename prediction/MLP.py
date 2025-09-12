import math
import torch
import torch.nn as nn
import sys
sys.path.append(sys.path[0] + '/..')
from prediction.BaseTrainableModel import BaseTrainableModel
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
    def __init__(self, in_dim: int, out_dim: int):
        """Initialise a TrainableMLP instance."""
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        dimensions = [self.in_dim] + CM().get('mlp.hidden_dims') + [self.out_dim]
        layers = []
        for i in range(len(dimensions) - 1):
            layers.append(nn.Linear(dimensions[i], dimensions[i + 1], device = CM().get('device')))
            layers.append(nn.ReLU())
        self.net = nn.ModuleList(layers)

    def forward(self, x):
        """Propagate input through the model."""
        for layer in self.net:
            x = layer(x)
        return x + torch.abs(x)

    def get_output_size(self):
        """Return output size of the network."""
        return self.out_dim

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
        self.flattened_in_dim = math.prod(self.in_dim)
        self.flattened_out_dim = math.prod(self.out_dim)

        return TrainableMLP(self.flattened_in_dim, self.flattened_out_dim)

    def get_model_output(self, src, tgt = None):
        """
        Get output of the model for given input.

        For the MLP architecture, only src is needed. tgt and guidance are leftovers from superclass signature and not used.

        Args:
            src: Input data.
            tgt: Target data. Ignore.

        Returns:
            Output of the model.
        """
        src = src.reshape(-1, self.flattened_in_dim)
        out = self.model(src)
        return out.reshape(-1, self.out_dim[0], self.out_dim[1])

    def scale_gradients(self):
        if self.guidance == "free":
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

    def get_architecture_name(self):
        """
        Return name of model architecture.
        """
        return "mlp"

