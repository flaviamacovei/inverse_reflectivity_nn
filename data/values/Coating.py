import torch

class Coating():

    def __init__(self, thicknesses: torch.Tensor):
        self.thicknesses = thicknesses

    def get_thicknesses(self):
        return self.thicknesses