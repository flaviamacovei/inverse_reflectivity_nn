import torch
from abc import ABC, abstractmethod

class BaseReflectiveProps(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def to(self, device: str):
        pass