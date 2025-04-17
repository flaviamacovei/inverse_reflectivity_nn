import torch
import sys
sys.path.append(sys.path[0] + '/..')
from data.values.Coating import Coating

class TreeNode:
    """
    Tree node class for branch and bound tree.

    Attributes:
        coating: Coating object.
        error: loss value for coating.
        lower_bound: lower bound value for coating.
        upper_bound: upper bound value for coating.

    Methods:
        get_coating: return coating object.
        get_error: return error value.
        get_lower_bound: return lower bound.
        get_upper_bound: return upper bound.
    """
    def __init__(self, coating: Coating, error: float, lower_bound: torch.Tensor, upper_bound: torch.Tensor):
        """
        Initialise a TreeNode instance.

        Args:
            coating: Coating object.
            error: loss value for coating.
            lower_bound: lower bound value for coating.
            upper_bound: upper bound value for coating.
        """
        self.coating = coating
        self.error = error
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def get_coating(self):
        """Return coating object."""
        return self.coating

    def get_error(self):
        """Return error value."""
        return self.error

    def get_lower_bound(self):
        """Return lower bound."""
        return self.lower_bound

    def get_upper_bound(self):
        """Return upper bound."""
        return self.upper_bound

    def __str__(self):
        """Return string representation of object."""
        return f"TreeNode(\n\terror={self.error}, \n\tlower_bound={self.lower_bound.squeeze().cpu().detach().numpy()}, \n\tupper_bound={self.upper_bound.squeeze().cpu().detach().numpy()}\n)"