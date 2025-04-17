from abc import ABC, abstractmethod
import sys
sys.path.append(sys.path[0] + '/..')
from data.values.ReflectivePropsPattern import ReflectivePropsPattern

class BaseModel(ABC):
    """
    Abstract base class for prediction models.

    This class provides a common interface for predicting coatings.
    It is intended to be subclassed by specific models with extended functionality.

    Methods:
        predict: Predict a coating given a reflective properties pattern object. Must be implemented by subclasses.
    """

    def __init__(self):
        """Initialise a BaseModel instance."""
        pass
    @abstractmethod
    def predict(self, target: ReflectivePropsPattern):
        """
        Predict a coating given a reflective properties pattern object. Must be implemented by subclasses.

        Args:
            target: Reflective properties pattern for which to perform prediction.
        """
        pass
