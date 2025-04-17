import torch
from abc import ABC, abstractmethod

class BaseReflectiveProps(ABC):
    """
    Abstract Base class for reflective properties.

    This class provides a common interface for modelling reflective properties.
    It is intended to be subclassed by reflective properties classes of type pattern and value.

    Methods:
        to: Move property tensors to device. Must be implemented by subclasses.
    """
    def __init__(self):
        """Initialise a BaseReflectiveProps instance."""
        pass

    @abstractmethod
    def to(self, device: str):
        """Move property tensors to device. Must be implemented by subclasses."""
        pass