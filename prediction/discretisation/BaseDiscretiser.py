from abc import ABC, abstractmethod
import sys
sys.path.append(sys.path[0] + '/..')
from prediction.relaxation.BaseRelaxedSolver import BaseRelaxedSolver
from data.values.ReflectivePropsPattern import ReflectivePropsPattern

class BaseDiscretiser(ABC):
    def __init__(self, relaxed_solver: BaseRelaxedSolver):
        self.relaxed_solver = relaxed_solver

    @abstractmethod
    def solve_discretised(self, target: ReflectivePropsPattern):
        pass
