from abc import ABC, abstractmethod
import torch
import sys
sys.path.append(sys.path[0] + '/..')
from prediction.relaxation.BaseRelaxedSolver import BaseRelaxedSolver
from data.values.ReflectivePropsPattern import ReflectivePropsPattern
from data.values.RefractiveIndex import RefractiveIndex
from utils.ConfigManager import ConfigManager as CM

class BaseDiscretiser(ABC):
    def __init__(self, relaxed_solver: BaseRelaxedSolver):
        self.relaxed_solver = relaxed_solver

    @abstractmethod
    def predict(self, target: ReflectivePropsPattern):
        pass

    def get_nondiscrete_map(self, input: torch.Tensor):
        assert len(input.shape) == 2
        rounded_tensor = RefractiveIndex.round_tensor(input)
        nondiscrete_map = torch.zeros(input.shape, dtype = torch.int64, device = CM().get('device'))
        for i in range(input.shape[0]):
            for j in range(input.shape[1]):
                if input[i][j] != rounded_tensor[i][j]:
                    nondiscrete_map[i][j] = 1
        return nondiscrete_map
