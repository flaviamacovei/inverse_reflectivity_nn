import torch

class ReflectiveProps:
    def __init__(self, start_wl: int, end_wl: int, properties: torch.Tensor):
        self.start_wl = start_wl
        self.end_wl = end_wl
        self.steps = properties.shape[0]
        self.properties = properties

    def get_start_wl(self):
        return self.start_wl

    def get_end_wl(self):
        return self.end_wl

    def get_steps(self):
        return self.steps

    def get_properties(self):
        return self.properties