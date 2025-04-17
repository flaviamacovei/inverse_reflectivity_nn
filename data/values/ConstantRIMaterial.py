import torch
from typing import Union
import sys
sys.path.append(sys.path[0] + '/..')
from data.values.Material import Material
from utils.ConfigManager import ConfigManager as CM

class ConstantRIMaterial(Material):
    """
    Material class for representing materials with single-value refractive index.

    Attributes:
        title: Name of the material.
        refractive_index: Tensor representing the refractive index of the material. Shape: (1)

    Methods:
        get_refractive_indices(): Return the refractive index of the material for each wavelength.
        get_coeffs(): Return the material's physical coefficients.
    """
    def __init__(self, title: str, reflective_index: Union[torch.Tensor, float]):
        """
        Initialise a ConstantRIMaterial instance.

        Args:
            title: Name of the material.
            reflective_index: Tensor or float representing the refractive index of the material.
        """
        super().__init__(title)
        self.refractive_index = reflective_index if isinstance(reflective_index, torch.Tensor) else torch.tensor([reflective_index], device = CM().get('device'))

    def get_refractive_indices(self):
        """Return the refractive index of the material for each wavelength."""
        result = torch.zeros_like(CM().get('wavelengths'))
        result[:] = self.refractive_index
        return result

    def get_coeffs(self):
        """Return the material's physical coefficients."""
        return self.refractive_index

    def __str__(self):
        """Return string representation of object."""
        return f"{self.title}:\nrefractive index: {self.refractive_index.cpu().detach().numpy()}"
