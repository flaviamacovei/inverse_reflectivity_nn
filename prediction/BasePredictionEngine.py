from abc import ABC, abstractmethod
import sys
sys.path.append(sys.path[0] + '/..')
from prediction.relaxation.BaseRelaxedSolver import BaseRelaxedSolver
from prediction.discretisation.BaseDiscretiser import BaseDiscretiser
from data.values.ReflectivePropsPattern import ReflectivePropsPattern

class BasePredictionEngine(ABC):
    def __init__(self):
        self.relaxed_solver: BaseRelaxedSolver = None
        self.discretiser: BaseDiscretiser = None

    def predict(self, target: ReflectivePropsPattern):
        return self.discretiser.solve_discretised(target)