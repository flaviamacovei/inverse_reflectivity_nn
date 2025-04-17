from prediction.BaseModel import BaseModel
import torch
import sys
sys.path.append(sys.path[0] + '/..')
from data.values.ReflectivePropsPattern import ReflectivePropsPattern
from data.values.Coating import Coating

class RandomModel(BaseModel):
    """
    Prediction model performing random prediction.

    Attributes:
        num_layers: |coating| to predict.
    """
    def __init__(self):
        """Initialise a RandomModel instance."""
        super().__init__()
        self.num_layers = CM().get('num_layers')

    def predict(self, target: ReflectivePropsPattern):
        # TODO: update or remove
        thicknesses_tensor = torch.rand((self.num_layers))
        refractive_indices_tensor = torch.rand((self.num_layers))
        return Coating(thicknesses_tensor, refractive_indices_tensor)