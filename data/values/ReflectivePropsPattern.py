import torch
from data.values.BaseReflectiveProps import BaseReflectiveProps

class ReflectivePropsPattern(BaseReflectiveProps):
    def __init__(self, lower_bound: torch.Tensor, upper_bound: torch.Tensor):
        assert lower_bound.shape == upper_bound.shape
        assert lower_bound.device == upper_bound.device
        assert torch.all(lower_bound <= upper_bound), f"Lower bound must be less than or equal to upper bound: issue at index {torch.argmax(lower_bound - upper_bound)} (lower bound: {lower_bound[:, torch.argmax(lower_bound - upper_bound)].squeeze().cpu().detach().numpy()}, upper bound: {upper_bound[:, torch.argmax(lower_bound - upper_bound)].squeeze().cpu().detach().numpy()})"
        super().__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def get_lower_bound(self):
        return self.lower_bound

    def get_upper_bound(self):
        return self.upper_bound

    def to(self, device: str):
        return ReflectivePropsPattern(self.lower_bound.to(device), self.upper_bound.to(device))

    def get_device(self):
        return self.lower_bound.device

    def __eq__(self, other):
        if isinstance(other, ReflectivePropsPattern):
            return torch.equal(self.lower_bound, other.get_lower_bound()) and torch.equal(self.upper_bound, other.get_upper_bound())
        return False

    def __str__(self):
        return f"Reflective Props Pattern object:\n\tlower bound: {self.lower_bound.squeeze().cpu().detach().numpy()},\n\tupper bound: {self.upper_bound.squeeze().cpu().detach().numpy()}"