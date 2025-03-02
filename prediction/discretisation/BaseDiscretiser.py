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
    def solve(self, target: ReflectivePropsPattern):
        pass

    def get_nondiscrete_indices(self, input: torch.Tensor):
        assert len(input.shape) == 2
        input = input.flatten()
        rounded_tensor = RefractiveIndex.round_tensor(input).flatten()
        nondiscrete_indices = torch.zeros(input.shape, dtype = torch.int64, device = CM().get('device')).flatten()
        for i in range(input.shape[0]):
            if input[i] != rounded_tensor[i]:
                nondiscrete_indices[i] = 1
        return nondiscrete_indices.nonzero(as_tuple = True)[0].tolist()
