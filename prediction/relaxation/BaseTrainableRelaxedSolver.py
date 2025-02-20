from abc import ABC, abstractmethod
import sys
sys.path.append(sys.path[0] + '/..')
from data.dataloaders.BaseDataloader import BaseDataloader
from prediction.relaxation.BaseRelaxedSolver import BaseRelaxedSolver

class BaseTrainableRelaxedSolver(BaseRelaxedSolver, ABC):
    def __init__(self, dataloader: BaseDataloader):
        super().__init__()
        self.dataloader = dataloader

    @abstractmethod
    def train(self):
        pass