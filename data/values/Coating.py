import torch

class Coating():

    def __init__(self, thicknesses: torch.Tensor, refractive_indices: torch.Tensor):
        assert thicknesses.shape == refractive_indices.shape
        assert thicknesses.device == refractive_indices.device
        self.thicknesses = thicknesses
        self.refractive_indices = refractive_indices

    def get_thicknesses(self):
        return self.thicknesses

    def get_refractive_indices(self):
        return self.refractive_indices

    def __str__(self):
        return f"Coating object:\n\tthicknesses: {self.thicknesses.squeeze().cpu().detach().numpy()},\n\trefractive_indices: {self.refractive_indices.squeeze().cpu().detach().numpy()}"

    def to(self, device: str):
        return Coating(self.thicknesses.to(device), self.refractive_indices.to(device))

    def device(self):
        return self.thicknesses.device

    def __eq__(self, other):
        if isinstance(other, Coating):
            return torch.equal(self.thicknesses, other.get_thicknesses()) and torch.equal(self.refractive_indices, other.get_refractive_indices())
        return False