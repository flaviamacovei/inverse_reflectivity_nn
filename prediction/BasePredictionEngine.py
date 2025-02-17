from abc import ABC, abstractmethod
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import sys
sys.path.append(sys.path[0] + '/..')
from data.values.ReflectivePropsPattern import ReflectivePropsPattern
from data.values.ReflectivePropsValue import ReflectivePropsValue
from config import wavelengths

class BasePredictionEngine(ABC):

    def __init__(self):
        pass
    @abstractmethod
    def predict(self, target: ReflectivePropsPattern):
        pass
