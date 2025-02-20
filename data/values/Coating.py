import torch

class Coating():

    def __init__(self, thicknesses: torch.Tensor, refractive_indices: torch.Tensor):
        self.thicknesses = thicknesses
        self.refractive_indices = refractive_indices

    def get_thicknesses(self):
        return self.thicknesses

    def get_refractive_indices(self):
        return self.refractive_indices

    def __str__(self):
        return f"Coating object:\n\tthicknesses: {self.thicknesses.detach().numpy()},\n\trefractive_indices: {self.refractive_indices.detach().numpy()}"