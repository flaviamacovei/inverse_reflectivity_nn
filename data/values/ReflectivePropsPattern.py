import torch
from data.values.BaseReflectiveProps import BaseReflectiveProps

class ReflectivePropsPattern(BaseReflectiveProps):
    def __init__(self, lower_bound: torch.Tensor, upper_bound: torch.Tensor):
        assert lower_bound.shape == upper_bound.shape
        assert torch.all(lower_bound <= upper_bound)
        super().__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def get_lower_bound(self):
        return self.lower_bound

    def get_upper_bound(self):
        return self.upper_bound

    def to(self, device = str):
        return ReflectivePropsPattern(self.lower_bound.to(device), self.upper_bound.to(device))