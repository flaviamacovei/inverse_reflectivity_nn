from data.BaseDataloader import BaseDataloader
import torch
from values.ReflectiveProps import ReflectiveProps
from values.Coating import Coating

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

    def load_data(self):
        self.dataset = list()
        for _ in range(self.num_points):
            properties_tensor = torch.rand((self.STEPS, 2))
            reflective_props = ReflectiveProps(self.START_WL, self.END_WL, properties_tensor)
            thicknesses_tensor = (self.MAX_THICKNESS - self.MIN_THICKNESS) * torch.rand((self.num_layers)) + self.MIN_THICKNESS
            thicknesses_tensor[0] = float("Inf")
            thicknesses_tensor[-1] = float("Inf")
            coating = Coating(thicknesses_tensor)
            self.dataset.append((reflective_props, coating))