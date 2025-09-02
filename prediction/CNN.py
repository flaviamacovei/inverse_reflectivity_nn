import torch
import torch.nn as nn
import math
import sys
sys.path.append(sys.path[0] + '/..')
from prediction.BaseTrainableModel import BaseTrainableModel
from utils.ConfigManager import ConfigManager as CM

class TrainableCNN(nn.Module):
    """
    Trainable convolutional neural network. Extends nn.Module.

    Architecture:

                  1         x     2     x          |wavelengths|         -->
            hidden_channels x     2     x |wavelengths| / pooling_size^1 -->
                                          ...                            -->
               embed_dim    x |coating| x                1

    See readme for details.

    Attributes:
        num_layers: Number of layers in a coating, |coating|
        embed_dim: Embedding dimension of a coating layer = materials_embed_dim + 1
        channel_dims: Sequence of number of channels for convolution layers
        num_blocks: Number of convolution layers. Derived from channel_dims
        out_dim: Output size of the network. Equal to (num_layers, embed_dim)
        kernel_size: Kernel size of convolution layers
        stride: Stride for convolution layers
        padding: Size of 0-padding of convolutions for maintaining data size. Derived from kernel_size

    Methods:
        forward: Propagate input through the model.
        get_output_size: Return output size of the network.
    """
    def __init__(self):
        """Initialise a TrainableCNN instance."""
        super().__init__()
        self.num_layers = CM().get('num_layers') + 2
        self.embed_dim = CM().get('material_embedding.dim') + 1
        self.channel_dims = [1] + CM().get('cnn.channel_dims') + [self.embed_dim]
        self.num_blocks = len(self.channel_dims) - 1 # each convolution connects one channel dim to the next
        self.out_dim = [self.num_layers, self.embed_dim]
        self.kernel_size = CM().get('cnn.kernel_size')
        self.stride = 1
        self.padding = self.kernel_size // 2

        start_width = CM().get('wavelengths').shape[0]
        end_width = self.num_layers
        width_pooling = max(1, math.floor((start_width / end_width) ** (1/self.num_blocks)))
        self.pooling_size = [1, # 1 for the height dimension
                        width_pooling # floor(k-th root of |wl|/|c|) for width dimension
                        ]

        conv_blocks = []
        for i in range(self.num_blocks):
            block = nn.Sequential(
                nn.Conv2d(in_channels = self.channel_dims[i], out_channels = self.channel_dims[i + 1], kernel_size = self.kernel_size, stride = self.stride, padding = self.padding),
                nn.MaxPool2d(kernel_size = self.pooling_size),
            )
            conv_blocks.append(block)
        self.convolutions = nn.ModuleList(conv_blocks)

        conv_width = math.floor(start_width / (width_pooling ** self.num_blocks))
        self.linear = nn.Linear(conv_width, end_width)

        self.final_pool = nn.MaxPool2d(kernel_size = (2, 1))


    def forward(self, x):
        """Propagate input through the model."""
        lower_bound, upper_bound = torch.chunk(x, 2, 1)
        x = torch.stack([lower_bound, upper_bound], dim = 1)[:, None]  # (batch, 2 * |wl|) --> (batch, in_ch = 1, 2, |wl|)
        for layer in self.convolutions:
            x = layer(x)
        x = self.linear(x)
        x = self.final_pool(x)
        return x + torch.abs(x)

    def get_output_size(self):
        """Return output size of the model."""
        return self.out_dim

class CNN(BaseTrainableModel):
    """
    Trainable prediction model using a CNN as base.

    Attributes:
        model: Instance of TrainableCNN.
    """
    def __init__(self):
        """Initialise a CNN instance."""
        super().__init__(TrainableCNN().to(CM().get('device')))

    def get_model_output(self, src, tgt = None):
        """
        Get output of the model for given input.

        For the CNN architecture, only src is needed. tgt and guidance are leftovers from superclass signature and not used.

        Args:
            src: Input data.
            tgt: Target data. Ignore.

        Returns:
            Output of the model.
        """
        return self.model(src)

    def scale_gradients(self):
        pass

    def get_architecture_name(self):
        """
        Return name of model architecture.
        """
        return "cnn"

