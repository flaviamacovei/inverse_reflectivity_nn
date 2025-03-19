from abc import ABC, abstractmethod
import torch
from typing import Union
import sys
sys.path.append(sys.path[0] + '/..')
from utils.ConfigManager import ConfigManager as CM

class Material(ABC):
    def __init__(self, title: str):
        self.title = title

    @abstractmethod
    def get_refractive_indices(self):
        pass

    @abstractmethod
    def get_coeffs(self):
        pass

    def get_title(self):
        return self.title

    def __repr__(self):
        return self.title

    @abstractmethod
    def __str__(self):
        pass

    def __eq__(self, other):
        return self.title == other.get_title() and torch.all(self.get_coeffs() == other.get_coeffs())

    @abstractmethod
    def __hash__(self):
        pass

    def __lt__(self, other):
        return self.title < other.get_title()
