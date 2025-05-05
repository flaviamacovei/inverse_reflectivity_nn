import torch
import torch.nn as nn
import math
import sys
sys.path.append(sys.path[0] + '/..')
from prediction.BaseTrainableModel import BaseTrainableModel
from data.dataloaders.BaseDataloader import BaseDataloader
from utils.ConfigManager import ConfigManager as CM

class TrainableCNN(nn.Module):
    """
    Trainable convolutional neural network. Extends nn.Module.

    Architecture:
                  1         x 2 x     |wavelengths|    --> output_size // 2**3 x 2 x |wavelengths| // 5
        output_size // 2**3 x 2 x  |wavelengths| // 5  --> output_size // 2**2 x 2 x |wavelengths| // 25
        output_size // 2**2 x 2 x |wavelengths| // 25  --> output_size // 2**1 x 2 x |wavelengths| // 125
        output_size // 2**1 x 2 x |wavelengths| // 125 -->     output_size     x 2 x           1
             output_size    x 2 x           1          -->     output_size     x 1 x           1
    See readme for details.

    Attributes:
        output_size: Output size of the network. Equal to |coating| * (embedding_dim + 1)

    Methods:
        forward: Propagate input through the model.
        get_output_size: Return output size of the network.
    """
    def __init__(self):
        """Initialise a TrainableCNN instance."""
        super().__init__()
        pooling_size = (1, math.ceil(math.log(CM().get('wavelengths').size()[0], 5)))
        # output size = |coating| * (embedding_dim + 1)
        self.output_size = (CM().get('layers.max') + 2) * (CM().get('material_embedding.dim') + 1)
        #           1         x 2 x     |wavelengths|    --> output_size // 2**3 x 2 x |wavelengths| // 5
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = self.output_size // 2**3, kernel_size = 3, stride = 1, padding = 1),
            nn.MaxPool2d(kernel_size = pooling_size)
        )
        # output_size // 2**3 x 2 x  |wavelengths| // 5  --> output_size // 2**2 x 2 x |wavelengths| // 25
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels = self.output_size // 2**3, out_channels = self.output_size // 2**2, kernel_size = 3, stride = 1, padding = 1),
            nn.MaxPool2d(kernel_size = pooling_size)
        )
        # output_size // 2**2 x 2 x |wavelengths| // 25  --> output_size // 2**1 x 2 x |wavelengths| // 125
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels = self.output_size // 2**2, out_channels = self.output_size // 2, kernel_size = 3, stride = 1, padding = 1),
            nn.MaxPool2d(kernel_size = pooling_size)
        )
        # output_size // 2**1 x 2 x |wavelengths| // 125 -->     output_size     x 2 x           1
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(in_channels = self.output_size // 2, out_channels = self.output_size, kernel_size = 3, stride = 1, padding = 1),
            nn.MaxPool2d(kernel_size = pooling_size)
        )
        #      output_size    x 2 x           1          -->     output_size     x 1 x           1
        self.final_pool = nn.MaxPool2d(kernel_size = (2, 1))


    def forward(self, x):
        """Propagate input through the model."""
        x = x.view((x.shape[0], 1, 2, x.shape[1] // 2))
        out = self.conv_block_1(x)
        out = self.conv_block_2(out)
        out = self.conv_block_3(out)
        out = self.conv_block_4(out)
        out = self.final_pool(out)
        out = out.view((out.shape[0], out.shape[1]))
        return torch.abs(out)

    def get_output_size(self):
        """Return output size of the model."""
        return self.output_size

class CNN(BaseTrainableModel):
    """
    Trainable prediction model using a CNN as base.

    Attributes:
        model: Instance of TrainableCNN.
    """
    def __init__(self):
        """Initialise a CNN instance."""
        super().__init__(TrainableCNN().to(CM().get('device')))

    def get_model_output(self, src, tgt = None, guidance = 'free'):
        """
        Get output of the model for given input.

        For the CNN architecture, only src is needed. tgt and guidance are leftovers from superclass signature and not used.

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
