import torch
import sys
sys.path.append(sys.path[0] + '/..')
from data.values.Coating import Coating

class TreeNode:
    def __init__(self, coating: Coating, error: float, lower_bound: torch.Tensor, upper_bound: torch.Tensor):
        self.coating = coating
        self.error = error
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def get_coating(self):
        return self.coating

    def get_error(self):
        return self.error

    def get_lower_bound(self):
        return self.lower_bound

    def get_upper_bound(self):
        return self.upper_bound

    def __str__(self):
        return f"TreeNode(\n\terror={self.error}, \n\tlower_bound={self.lower_bound.squeeze().cpu().detach().numpy()}, \n\tupper_bound={self.upper_bound.squeeze().cpu().detach().numpy()}\n)"