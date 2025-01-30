from DataloaderInterface import BaseDataloader
import random
import torch

class RandomDataloader(BaseDataloader):

    def __init__(self, batch_size: int, num_wavelengths: int, num_layers: int, shuffle: bool = False, num_points: int = 1000):
        super().__init__(batch_size = batch_size, num_wavelengths = num_wavelengths, num_layers = num_layers, shuffle = shuffle)
        self.num_points = num_points
        self.MIN_THICKNESS = 1.0e-08
        self.MAX_THICKNESS = 1.0e-07

    def load_data(self):
        self.dataset = list()
        for _ in range(self.num_points):
            properties = torch.complex(torch.rand((self.num_wavelengths, 2)), torch.rand((self.num_wavelengths, 2)))
            thicknesses = (self.MAX_THICKNESS - self.MIN_THICKNESS) * torch.rand((self.num_layers)) + self.MIN_THICKNESS
            thicknesses[0] = float("Inf")
            thicknesses[-1] = float("Inf")
            self.dataset.append((properties, thicknesses))