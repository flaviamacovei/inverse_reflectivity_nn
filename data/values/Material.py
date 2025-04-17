from abc import ABC, abstractmethod
import torch
from typing import Union
import sys
sys.path.append(sys.path[0] + '/..')
from utils.ConfigManager import ConfigManager as CM
from utils.os_utils import short_hash

class Material(ABC):
    """
    Abstract base class for materials.

    This class provides a common interface for handling the physical properties of materials.
    It is intended to be subclassed by specific material classes.

    Attributes:
        title: Name of the material.

    Methods:
        get_refractive_indices(): Return the refractive index of the material for each wavelength.
        get_coeffs(): Return the material's physical coefficients.
    """
    def __init__(self, title: str):
        """
        Initialise a Material instance.

        Args:
            title: Name of the material.
        """
        self.title = title

    @abstractmethod
    def get_refractive_indices(self):
        """Return the refractive index of the material for each wavelength. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def get_coeffs(self):
        """Return the material's physical coefficients. Must be implemented by subclasses."""
        pass

    def get_title(self):
        """Return the name of the material."""
        return self.title

    def __repr__(self):
        """Return string representation of object for logging."""
        return self.title

    @abstractmethod
    def __str__(self):
        """Return string representation of object. Must be implemented by subclasses."""
        pass

    def __eq__(self, other):
        """
        Compare this Material object with other object.

        Args:
            other: Object with which to compare.

        Returns:
            True if other is Material object with the same title and physical coefficients.
        """
        return self.title == other.get_title() and torch.all(self.get_coeffs() == other.get_coeffs())

    def __hash__(self):
        """Return hash value of object."""
        return short_hash(self)

    def __lt__(self, other):
        """
        Compare this Material object with other object.

        Args:
            other: Object with which to compare.

        Returns:
            True if other is Material object with alphabetically later title.
        """
        return self.title < other.get_title()
