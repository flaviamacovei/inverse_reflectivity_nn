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
        2 x |wavelengths| --> 128 --> 512 --> 64 --> output_size
    See readme for details.

    Attributes:
        output_size: Output size of the network. Equal to |coating| * (embedding_dim + 1)

    Methods:
        forward: Propagate input through the model.
        get_output_size: Return output size of the network.
    """
    def __init__(self):
        """Initialise a TrainableMLP instance."""
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
        """Propagate input through the model."""
        return torch.abs(self.net(x))

    def get_output_size(self):
        """Return output size of the network."""
        return self.output_size

class MLP(BaseTrainableModel):
    """
    Trainable prediction model using an MLP as base.

    Attributes:
        model: Instance of TrainableMLP.
    """
    def __init__(self):
        """Initialise an MLP instance."""
        super().__init__(TrainableMLP().to(CM().get('device')))

    def get_model_output(self, src, tgt = None, guidance = 'free'):
        """
        Get output of the model for given input.

        For the MLP architecture, only src is needed. tgt and guidance are leftovers from superclass signature and not used.

        Args:
            src: Input data.
            tgt: Target data. Ignore.
            guidance: Guidance data. Ignore.

        Returns:
            Output of the model.
        """
        return self.model(src)

    def scale_gradients(self):
        pass
