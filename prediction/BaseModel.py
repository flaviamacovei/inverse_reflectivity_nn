from abc import ABC, abstractmethod
import sys
sys.path.append(sys.path[0] + '/..')
from data.values.ReflectivePropsPattern import ReflectivePropsPattern

class BaseModel(ABC):

    def __init__(self):
        pass
    @abstractmethod
    def predict(self, target: ReflectivePropsPattern):
        pass
