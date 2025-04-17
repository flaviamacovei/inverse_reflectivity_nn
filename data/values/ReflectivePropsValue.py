import torch
from data.values.BaseReflectiveProps import BaseReflectiveProps

class ReflectivePropsValue(BaseReflectiveProps):
    """
    Reflective properties value class for modelling reflective properties using value tensor.

    This class is intended for comparison with a ReflectivePropsPattern instance.
    A value matches a pattern if at every wavelength, the value lies between the pattern lower and upper bound.

    Attributes:
        value: Tensor representation of value. Shape: (batch_size, |wavelengths|)

    Methods:
        get_value: Return value.
        to: Move property tensor to device.
        get_device: Return device of ReflectivePropsValue object.
    """

    def __init__(self, value: torch.Tensor):
        """
        Initialise a ReflectivePropsValue instance.

        Args:
            value: Tensor representation of value. Shape: (batch_size, |wavelengths|)
        """
        assert len(value.shape) == 2
        super().__init__()
        self.value = value

    def get_value(self):
        """Return value."""
        return self.value

    def to(self, device: str):
        """Move property tensor to device."""
        return ReflectivePropsValue(self.value.to(device))

    def get_device(self):
        """Return device of ReflectivePropsValue object."""
        return self.value.device

    def __eq__(self, other):
        """
        Compare this ReflectivePropsValue object with other object.

        Args:
            other: Object with which to compare.

        Returns:
            True if other is ReflectivePropsPattern object with same lower and upper bound.
        """
        if isisntance(other, ReflectivePropsValue):
            return torch.equal(self.value, other.get_value())
        return False

    def __str__(self):
        """Return string representation of ReflectivePropsValue object."""
        return f"Reflective Props Value object:\n\tvalue: {self.value.squeeze().cpu().detach().numpy()}"