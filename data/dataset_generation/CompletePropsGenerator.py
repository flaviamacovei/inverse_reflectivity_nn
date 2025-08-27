import torch
import sys
sys.path.append(sys.path[0] + '/../..')
from data.dataset_generation.BaseGenerator import BaseGenerator
from data.values.Coating import Coating
from data.values.ReflectivityPattern import ReflectivityPattern
from forward.forward_tmm import coating_to_reflectivity
from data.material_embedding.EmbeddingManager import EmbeddingManager as EM

class CompletePropsGenerator(BaseGenerator):
    """
    Complete Properties Generator class for generating datasets of density 'complete'.

    A 'complete' point contains reflectivity information at every wavelength.
    Lower bound and upper bound differ by no more than config.tolerance.
    See readme for more information and examples.

    Methods:
        make_point: Generate a 'complete' point.
    """
    def __init__(self, num_points = 1, batch_size: int = 256):
        """Initialise a CompletePropsGenerator instance."""
        super().__init__(num_points, batch_size)

    def make_points(self, num_points: int):
        """
        Generate 'complete' points.

        Returns:
            pattern: ReflectivityPattern instance where lower bound and upper bound differ by no more than config.tolerance.
            coating: corresponding Coating instance.
        """
        materials_indices = self.make_materials_choice(num_points)
        thicknesses = self.make_thicknesses(materials_indices)

        # make features
        embedding = self.get_materials_embeddings(materials_indices)
        coating_encoding = torch.cat([thicknesses[:, :, None], embedding], dim = 2).float()
        coating = Coating(coating_encoding)

        # make labels
        reflectivity = coating_to_reflectivity(coating).get_value().float()

        lower_bound = torch.clamp(reflectivity - self.TOLERANCE / 2, 0, 1)
        upper_bound = torch.clamp(reflectivity + self.TOLERANCE / 2, 0, 1)

        pattern = ReflectivityPattern(lower_bound, upper_bound)

        return pattern, coating