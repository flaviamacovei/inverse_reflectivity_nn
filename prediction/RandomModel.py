import torch
import torch.nn.functional as F
import sys
sys.path.append(sys.path[0] + '/..')
from prediction.BaseModel import BaseModel
from data.values.ReflectivityPattern import ReflectivityPattern
from data.values.Coating import Coating

from data.dataset_generation.CompletePatternGenerator import CompletePatternGenerator

class RandomModel(BaseModel):
    """
    Prediction model performing random prediction.

    Attributes:
        num_layers: |coating| to predict.
    """
    def __init__(self):
        """Initialise a RandomModel instance."""
        super().__init__()
        self.generator = CompletePatternGenerator()

    def get_random_thicknesses(self, materials_indices):
        assert len(materials_indices.shape) == 2 or len(materials_indices.shape) == 3
        if len(materials_indices.shape) == 3:
            materials_indices = torch.argmax(materials_indices, dim = -1)
        return self.generator.make_thicknesses(materials_indices)

    def get_random_materials_hard(self, num_points):
        return self.generator.make_materials_choice(num_points)

    def get_random_materials_soft(self, num_points):
        return torch.rand(size = (num_points, self.tgt_seq_len, self.tgt_vocab_size)) * self.tgt_vocab_size

    def model_predict(self, target: ReflectivityPattern):
        num_points = target.get_batch_size()
        materials = self.get_random_materials_hard(num_points)
        thicknesses = self.get_random_thicknesses(materials)
        coating_encoding = torch.cat([thicknesses[:, :, None], materials], dim = -1).float()
        coating = Coating(coating_encoding)
        return coating

    def get_architecture_name(self):
        """
        Return name of model architecture.
        """
        return "random"
