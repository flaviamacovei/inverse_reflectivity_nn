from abc import ABC, abstractmethod
import sys
sys.path.append(sys.path[0] + '/..')
from data.values.ReflectivityPattern import ReflectivityPattern
from utils.ConfigManager import ConfigManager as CM

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
        # shape variables
        self.src_seq_len = CM().get('wavelengths').shape[0]
        self.src_dim = 2  # lower bound and upper bound
        self.tgt_seq_len = CM().get('num_layers') + 2  # thin films + substrate + air
        self.tgt_vocab_size = len(CM().get('materials.thin_films')) + 2  # available thin films + substrate + air
        self.tgt_dim = 2
        self.in_dims = {'seq_len': self.src_seq_len, 'dim': self.src_dim}
        self.out_dims = {'seq_len': self.tgt_seq_len, 'material': self.tgt_vocab_size, 'thickness': 1}

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
