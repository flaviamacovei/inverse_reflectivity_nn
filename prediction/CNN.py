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
        num_layers: Number of convolution layers. Derived from channel_dims
        out_dim: Output size of the network. Equal to (num_layers, embed_dim)
        kernel_size: Kernel size of convolution layers
        stride: Stride for convolution layers
        padding: Size of 0-padding of convolutions for maintaining data size. Derived from kernel_size

    Methods:
        forward: Propagate input through the model.
        get_output_size: Return output size of the network.
    """
    def __init__(self, in_dim: list[int], out_dim: list[int], channel_dims: list[int], kernel_size: int, stride: int, padding: int):
        """Initialise a TrainableCNN instance."""
        super().__init__()
        self.in_dim = in_dim
        src_width, src_height = in_dim
        self.out_dim = out_dim
        tgt_width, tgt_height = out_dim
        self.channel_dims = [1] + channel_dims + [1]
        self.num_layers = len(self.channel_dims) - 1 # each convolution connects one channel dim to the next
        # kernel size for (untransposed) convolutions
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # increase height with transpose convolution
        height_pooling = max(1, math.ceil((tgt_height - src_height) / self.num_layers) + 1) # why + 1? understand formula from https://www.youtube.com/watch?v=yPKHCGbmKZY
        self.transpose_kernel_size = [
            1, # 1 for width convolution
            height_pooling, # ceil(k-th root of (|vocab| - 1)/(2 - 1)) + 1 for height dimension
        ]

        # decrease width with max pooling
        width_pooling = max(1, math.floor((src_width / tgt_width) ** (1 / self.num_layers)))
        self.pooling_size = [
            width_pooling, # floor(k-th root of |wl|/|c|) for width dimension
            1, # 1 for height pooling
            ]

        conv_blocks = []
        for i in range(self.num_layers):
            block = nn.Sequential(
                nn.Conv2d(in_channels = self.channel_dims[i], out_channels = self.channel_dims[i + 1], kernel_size = self.kernel_size, stride = self.stride, padding = self.padding),
                nn.ConvTranspose2d(in_channels = self.channel_dims[i + 1], out_channels = self.channel_dims[i + 1], kernel_size = self.transpose_kernel_size, stride = self.stride, padding = 0),
                nn.MaxPool2d(kernel_size = self.pooling_size),
            )
            conv_blocks.append(block)
        self.convolutions = nn.ModuleList(conv_blocks)

        # achieved width after convolutions with rounded kernel
        self.conv_width = math.floor(src_width / (width_pooling ** self.num_layers))
        # achieved height after convolutions with rounded kernel
        self.conv_height = math.floor(src_height + (height_pooling - 1) * self.num_layers)
        self.linear = nn.Linear(self.conv_width * self.conv_height, tgt_width * tgt_height)

        self.final_pool = nn.MaxPool2d(kernel_size = (2, 1))


    def forward(self, x):
        """Propagate input through the model."""
        x = x[:, None] # add empty channel dim to input
        for layer in self.convolutions:
            x = layer(x)
        x = x.reshape(-1, self.conv_width * self.conv_height)
        x = self.linear(x)
        x = x.reshape(-1, self.out_dim[0], self.out_dim[1])
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
        super().__init__()


    def build_model(self):
        channel_dims = CM().get('cnn.channel_dims')
        kernel_size = CM().get('cnn.kernel_size')
        stride = 1
        padding = kernel_size // 2
        return TrainableCNN(self.in_dim, self.out_dim, channel_dims, kernel_size, stride, padding).to(CM().get('device'))

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

