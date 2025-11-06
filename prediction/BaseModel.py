from abc import ABC, abstractmethod
from typing import Union
import torch
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

    def predict(self, target: Union[ReflectivityPattern, torch.Tensor]):
        """
        Predict a coating given a reflectivity pattern object.

        Args:
            target: Reflectivity pattern for which to perform prediction.
        """
        assert type(target) in [ReflectivityPattern, torch.Tensor], "target must be either a ReflectivityPattern or a torch.Tensor"
        if type(target) == torch.Tensor:
            assert len(target.shape) == 2, f"target tensor must have batch and reflectivity dimensions, found {len(target.shape)}"
            lower_bound, upper_bound = target.chunk(2, -1)
            target = ReflectivityPattern(lower_bound, upper_bound)
        return self.model_predict(target)

    @abstractmethod
    def model_predict(self, target: ReflectivityPattern):
        """
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def get_architecture_name(self):
        """
        Return name of model architecture. Must be implemented by subclasses.
        """
        pass
