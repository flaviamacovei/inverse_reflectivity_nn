import torch
from data.values.BaseReflectivity import BaseReflectivity

class ReflectivityValue(BaseReflectivity):
    """
    Reflectivity value class for modelling reflectivity using value tensor.

    This class is intended for comparison with a ReflectivityPattern instance.
    A value matches a pattern if at every wavelength, the value lies between the pattern lower and upper bound.

    Attributes:
        value: Tensor representation of value. Shape: (batch_size, |wavelengths|)

    Methods:
        get_value: Return value.
        to: Move property tensor to device.
        get_device: Return device of ReflectivityValue object.
    """

    def __init__(self, value: torch.Tensor):
        """
        Initialise a ReflectivityValue instance.

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
        return ReflectivityValue(self.value.to(device))

    def get_device(self):
        """Return device of ReflectivityValue object."""
        return self.value.device

    def get_batch(self, i):
        return ReflectivityValue(self.value[i:i + 1])

    def get_batch_size(self):
        return self.value.shape[0]

    def __eq__(self, other):
        """
        Compare this ReflectivityValue object with other object.

        Args:
            other: Object with which to compare.

        Returns:
            True if other is ReflectivityValue object with same value.
        """
        if isinstance(other, ReflectivityValue):
            return torch.equal(self.value, other.get_value())
        return False


    def __str__(self):
        """Return string representation of ReflectivityValue object."""
        return f"Reflectivity Value object:\n\tvalue: {self.value.squeeze().cpu().detach().numpy()}"