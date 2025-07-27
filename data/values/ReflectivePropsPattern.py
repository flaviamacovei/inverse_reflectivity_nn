import torch
from data.values.BaseReflectiveProps import BaseReflectiveProps

class ReflectivePropsPattern(BaseReflectiveProps):
    """
    Reflective properties pattern class for modelling reflective properties using lower and upper bound.

    This class is intended for comparison with a ReflectivePropsValue instance.
    A value matches a pattern if at every wavelength, the value lies between the pattern lower and upper bound.

    Attributes:
        lower_bound: Tensor representation of lower bound. Shape: (batch_size, |wavelengths|)
        upper_bound: Tensor representation of upper bound. Shape: (batch_size, |wavelengths|)

    Methods:
        get_lower_bound: Return lower bound.
        get_upper_bound: Return upper bound.
        to: Move property tensors to device.
        get_device: Return device of ReflectivePropsPattern object.
    """

    def __init__(self, lower_bound: torch.Tensor, upper_bound: torch.Tensor):
        """
        Initialise a ReflectivePropsPattern instance.

        Args:
            lower_bound: Tensor representation of lower bound. Shape: (batch_size, |wavelengths|)
            upper_bound: Tensor representation of upper bound. Shape: (batch_size, |wavelengths|)
        """
        assert lower_bound.shape == upper_bound.shape
        assert lower_bound.device == upper_bound.device
        assert torch.all(lower_bound <= upper_bound), f"Lower bound must be less than or equal to upper bound: issue at index {torch.argmax(lower_bound - upper_bound)} (lower bound: {lower_bound[:, torch.argmax(lower_bound - upper_bound)].squeeze().cpu().detach().numpy()}, upper bound: {upper_bound[:, torch.argmax(lower_bound - upper_bound)].squeeze().cpu().detach().numpy()})"
        super().__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def get_lower_bound(self):
        """Return lower bound."""
        return self.lower_bound

    def get_upper_bound(self):
        """Return upper bound."""
        return self.upper_bound

    def get_batch_size(self):
        """Return batch size."""
        return self.lower_bound.shape[0]

    def to(self, device: str):
        """Move property tensors to device."""
        return ReflectivePropsPattern(self.lower_bound.to(device), self.upper_bound.to(device))

    def get_device(self):
        """Return device of ReflectivePropsPattern object."""
        return self.lower_bound.device

    def __eq__(self, other):
        """
        Compare this ReflectivePropsPattern object with other object.

        Args:
            other: Object with which to compare.

        Returns:
            True if other is ReflectivePropsPattern object with same lower and upper bound.
        """
        if isinstance(other, ReflectivePropsPattern):
            return torch.equal(self.lower_bound, other.get_lower_bound()) and torch.equal(self.upper_bound, other.get_upper_bound())
        return False

    def __str__(self):
        """Return string representation of ReflectivePropsPattern object."""
        return f"Reflective Props Pattern object:\n\tlower bound: {self.lower_bound.squeeze().cpu().detach().numpy()},\n\tupper bound: {self.upper_bound.squeeze().cpu().detach().numpy()}"