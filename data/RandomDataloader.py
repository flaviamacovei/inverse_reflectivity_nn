from data.BaseDataloader import BaseDataloader
from config import tolerance
import torch
from values.ReflectivePropsPattern import ReflectivePropsPattern
from values.Coating import Coating
from forward.forward_tmm import coating_to_reflective_props

class RandomDataloader(BaseDataloader):

    def __init__(self, batch_size: int, num_layers: int, shuffle: bool = False, num_points: int = 1000):
        super().__init__(batch_size = batch_size, shuffle = shuffle)
        self.num_layers = num_layers
        self.num_points = num_points
        self.MIN_THICKNESS = 1.0e-08
        self.MAX_THICKNESS = 1.0e-07
        self.START_WL = 500
        self.END_WL = 1500
        self.STEPS = 1000
        self.TOLERANCE = tolerance

    def load_data(self):
        self.dataset = list()
        for _ in range(self.num_points):
            thicknesses_tensor = (self.MAX_THICKNESS - self.MIN_THICKNESS) * torch.rand((self.num_layers)) + self.MIN_THICKNESS
            thicknesses_tensor[0] = float("Inf")
            thicknesses_tensor[-1] = float("Inf")
            coating = Coating(thicknesses_tensor)
            properties_tensor = coating_to_reflective_props(coating, self.START_WL, self.END_WL, self.STEPS)
            lower_bound = torch.clamp(properties_tensor - self.TOLERANCE / 2, 0, 1)
            upper_bound = torch.clamp(properties_tensor + self.TOLERANCE / 2, 0, 1)
            reflective_props = ReflectivePropsPattern(self.START_WL, self.END_WL, lower_bound, upper_bound)
            self.dataset.append((reflective_props, coating))