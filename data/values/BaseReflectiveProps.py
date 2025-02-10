import torch
from abc import ABC, abstractmethod

class BaseReflectiveProps(ABC):
    def __init__(self, start_wl: int, end_wl: int, steps: int):
        self.start_wl = start_wl
        self.end_wl = end_wl
        self.steps = steps

    def get_start_wl(self):
        return self.start_wl

    def get_end_wl(self):
        return self.end_wl

    def get_steps(self):
        return self.steps