import torch
from abc import ABC, abstractmethod

class BaseReflectivity(ABC):
    """
    Abstract Base class for reflectivity.

    This class provides a common interface for modelling reflectivity.
    It is intended to be subclassed by reflectivity classes of type pattern and value.

    Methods:
        to: Move property tensors to device. Must be implemented by subclasses.
    """
    def __init__(self):
        """Initialise a BaseReflectivity instance."""
        pass

    @abstractmethod
    def to(self, device: str):
        """Move property tensors to device. Must be implemented by subclasses."""
        pass