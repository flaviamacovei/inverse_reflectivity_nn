import torch
from typing import Union
import sys
sys.path.append(sys.path[0] + '/..')
from data.values.BaseMaterial import BaseMaterial
from utils.ConfigManager import ConfigManager as CM

class SellmeierMaterial(BaseMaterial):
    """
    Material class for representing materials with multiple-value Sellmeier coefficients.

    Attributes:
        title: Name of the material.
        B: Tensor representing Sellmeier coefficients B. Shape: (|B|).
        C: Tensor representing Sellmeier coefficients C. Shape: (|C|).

    Methods:
        get_refractive_indices(): Return the refractive index of the material for each wavelength.
        get_coeffs(): Return the material's physical coefficients.
    """
    def __init__(self, title: str, B: Union[torch.tensor, list[float]], C: Union[torch.tensor, list[float]]):
        """
        Initialise a SellmeierMaterial instance.

        Args:
            title: Name of the material.
            B: Tensor or list representing Sellmeier coefficients B. Shape: (|B|).
            C: Tensor or list representing Sellmeier coefficients C. Shape: (|C|).
        """
        assert len(B) == len(C)
        assert len(B) > 0
        super().__init__(title)
        self.B = B if isinstance(B, torch.Tensor) else torch.tensor(B, device = CM().get('device'))
        self.C = C if isinstance(C, torch.Tensor) else torch.tensor(C, device = CM().get('device'))

    def get_refractive_indices(self):
        """
        Return the refractive index of the material for each wavelength.

        Refractive index is calculated using the Sellmeier equation.

        Returns:
            Tensor representing the refractive indices. Shape: (|wavelengths|).
        """
        wl2 = CM().get('wavelengths') ** 2
        n2 = 1 + sum(b * wl2 / (wl2 - c) for b, c in zip(self.B, self.C))
        return n2 ** 0.5

    def get_coeffs(self):
        """Return the material's physical coefficients."""
        return torch.cat((self.B, self.C))

    def __str__(self):
        """Return string representation of object."""
        return f"{self.title}:\nB: {self.B.cpu().detach().numpy()}\nC: {self.C.cpu().detach().numpy()}"
