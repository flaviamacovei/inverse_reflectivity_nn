import torch
from data.values.BaseReflectiveProps import BaseReflectiveProps

class ReflectivePropsPattern(BaseReflectiveProps):
    def __init__(self, start_wl: int, end_wl: int, lower_bound: torch.Tensor, upper_bound: torch.Tensor):
        assert lower_bound.shape == upper_bound.shape
        assert torch.all(lower_bound <= upper_bound)
        super().__init__(start_wl, end_wl, lower_bound.shape[0])
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def get_lower_bound(self):
        return self.lower_bound

    def get_upper_bound(self):
        return self.upper_bound