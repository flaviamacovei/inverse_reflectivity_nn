from data.dataloaders.BaseDataloader import BaseDataloader
import torch
from data.values.Coating import Coating
from forward.forward_tmm import coating_to_reflective_props
from utils.ConfigManager import ConfigManager as CM


class RandomDataloader(BaseDataloader):

    def __init__(self, batch_size: int, num_layers: int, shuffle: bool = False, num_points: int = 1000):
        super().__init__(batch_size = batch_size, shuffle = shuffle)
        self.num_layers = num_layers
        self.num_points = num_points
        self.MIN_THICKNESS = 1.0e-08
        self.MAX_THICKNESS = 1.0e-07
        self.TOLERANCE = CM().get('tolerance')

    def load_data(self):
        self.dataset = list()
        for _ in range(self.num_points):
            thicknesses_tensor = (self.MAX_THICKNESS - self.MIN_THICKNESS) * torch.rand((self.num_layers)) + self.MIN_THICKNESS
            thicknesses_tensor[0] = float("Inf")
            thicknesses_tensor[-1] = float("Inf")
            refractive_indices_tensor = torch.rand((self.num_layers))
            coating = Coating(thicknesses_tensor, refractive_indices_tensor)
            properties_tensor = coating_to_reflective_props(coating).get_value()
            lower_bound = torch.clamp(properties_tensor - self.TOLERANCE / 2, 0, 1)
            upper_bound = torch.clamp(properties_tensor + self.TOLERANCE / 2, 0, 1)
            reflective_props_tensor = torch.cat((lower_bound, upper_bound))
            self.dataset.append(reflective_props_tensor)