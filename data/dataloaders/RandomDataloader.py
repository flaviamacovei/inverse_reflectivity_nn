from data.dataloaders.BaseDataloader import BaseDataloader
import torch
from data.values.Coating import Coating
from forward.forward_tmm import coating_to_reflectivity
from utils.ConfigManager import ConfigManager as CM


class RandomDataloader(BaseDataloader):
    """
    Random Dataloader class for experimentation.

    Methods:
        load_data: Load random data.
    """

    def __init__(self, batch_size: int, num_layers: int, shuffle: bool = False, num_points: int = 1000):
        """
        Initialise a RandomDataloader instance.

        Args:
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle the dataset before each epoch. Defaults to True.
            num_points: Number of samples to generate. Defaults to 1000.
        """
        super().__init__(batch_size = batch_size, shuffle = shuffle)
        self.num_layers = num_layers
        self.num_points = num_points
        self.MIN_THICKNESS = 1.0e-08
        self.MAX_THICKNESS = 1.0e-07
        self.TOLERANCE = CM().get('tolerance')

    def load_data(self):
        """Load random data."""
        self.dataset = list()
        for _ in range(self.num_points):
            # TODO: update
            # generate random thicknesses tensor
            thicknesses_tensor = (self.MAX_THICKNESS - self.MIN_THICKNESS) * torch.rand((self.num_layers)) + self.MIN_THICKNESS
            thicknesses_tensor[0] = float("Inf")
            thicknesses_tensor[-1] = float("Inf")
            # generate random refractive indices tensor
            refractive_indices = torch.rand((self.num_layers))
            coating = Coating(thicknesses_tensor, refractive_indices)
            # calculate reflectivity from coating
            reflectivity = coating_to_reflectivity(coating).get_value()
            lower_bound = torch.clamp(reflectivity - self.TOLERANCE / 2, 0, 1)
            upper_bound = torch.clamp(reflectivity + self.TOLERANCE / 2, 0, 1)
            reflectivity = torch.cat((lower_bound, upper_bound))
            self.dataset.append(reflectivity)