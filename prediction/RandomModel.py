from prediction.BaseModel import BaseModel
import torch
import sys
sys.path.append(sys.path[0] + '/..')
from data.values.ReflectivityPattern import ReflectivityPattern
from data.values.Coating import Coating
from utils.ConfigManager import ConfigManager as CM
from data.dataset_generation.CompletePropsGenerator import CompletePropsGenerator

class RandomModel(BaseModel):
    """
    Prediction model performing random prediction.

    Attributes:
        num_layers: |coating| to predict.
    """
    def __init__(self):
        """Initialise a RandomModel instance."""
        super().__init__()

    def predict(self, target: ReflectivityPattern):
        num_points = target.get_batch_size()

        # this isn't super pretty
        generator = CompletePropsGenerator()

        materials_indices = generator.make_materials_choice(num_points)
        thicknesses = generator.make_thicknesses(materials_indices)
        embedding = generator.get_materials_embeddings(materials_indices)
        coating_encoding = torch.cat([thicknesses[:, :, None], embedding], dim=2).float()
        coating = Coating(coating_encoding)

        return coating