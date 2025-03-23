import torch
import torch.nn as nn
import math
import sys
sys.path.append(sys.path[0] + '/..')
from prediction.BaseTrainableModel import BaseTrainableModel
from data.dataloaders.BaseDataloader import BaseDataloader
from utils.ConfigManager import ConfigManager as CM

class TrainableCNN(nn.Module):
    def __init__(self):
        super().__init__()
        pooling_size = (1, math.ceil(math.log(CM().get('wavelengths').size()[0], 5)))
        self.output_size = CM().get('layers.max') * (CM().get('material_embedding.dim') + 1)
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = self.output_size // 2**3, kernel_size = 3, stride = 1, padding = 1),
            nn.MaxPool2d(kernel_size = pooling_size)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels = self.output_size // 2**3, out_channels = self.output_size // 2**2, kernel_size = 3, stride = 1, padding = 1),
            nn.MaxPool2d(kernel_size = pooling_size)
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels = self.output_size // 2**2, out_channels = self.output_size // 2, kernel_size = 3, stride = 1, padding = 1),
            nn.MaxPool2d(kernel_size = pooling_size)
        )
        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(in_channels = self.output_size // 2, out_channels = self.output_size, kernel_size = 3, stride = 1, padding = 1),
            nn.MaxPool2d(kernel_size = pooling_size)
        )
        self.final_pool = nn.MaxPool2d(kernel_size = (2, 1))


    def forward(self, x):
        x = x.view((x.shape[0], 1, 2, x.shape[1] // 2))
        out = self.conv_block_1(x)
        out = self.conv_block_2(out)
        out = self.conv_block_3(out)
        out = self.conv_block_4(out)
        out = self.final_pool(out)
        out = out.view((out.shape[0], out.shape[1]))
        return torch.abs(out)

    def get_output_size(self):
        return self.output_size

class CNN(BaseTrainableModel):
    def __init__(self, dataloader: BaseDataloader):
        super().__init__(dataloader, TrainableCNN().to(CM().get('device')))

    def scale_gradients(self):
        pass
