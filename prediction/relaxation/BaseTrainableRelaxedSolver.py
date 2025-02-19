from abc import ABC, abstractmethod
from BaseRelaxedSolver import BaseRelaxedSolver
import sys
sys.path.append(sys.path[0] + '/..')
from data.BaseDataloader import BaseDataloader

class BaseTrainableRelaxedSolver(BaseRelaxedSolver, ABC):
    def __init__(self, dataloader: BaseDataloader):
        super().__init__()
        self.dataloader = dataloader

    @abstractmethod
    def train(self):
        pass