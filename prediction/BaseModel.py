from abc import ABC, abstractmethod
import sys
sys.path.append(sys.path[0] + '/..')
from data.values.ReflectivityPattern import ReflectivityPattern

class BaseModel(ABC):
    """
    Abstract base class for prediction models.

    This class provides a common interface for predicting coatings.
    It is intended to be subclassed by specific models with extended functionality.

    Methods:
        predict: Predict a coating given a reflectivity pattern object. Must be implemented by subclasses.
    """

    def __init__(self):
        """Initialise a BaseModel instance."""
        pass
    @abstractmethod
    def predict(self, target: ReflectivityPattern):
        """
        Predict a coating given a reflectivity pattern object. Must be implemented by subclasses.

        Args:
            target: Reflectivity pattern for which to perform prediction.
        """
        pass

    @abstractmethod
    def get_architecture_name(self):
        """
        Return name of model architecture. Must be implemented by subclasses.
        """
        pass
