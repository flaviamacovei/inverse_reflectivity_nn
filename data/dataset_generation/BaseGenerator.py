from abc import ABC, abstractmethod
import random
import torch
import numpy as np
import math
import sys

sys.path.append(sys.path[0] + '/../..')
from data.values.Coating import Coating
from utils.ConfigManager import ConfigManager as CM
from data.material_embedding.EmbeddingManager import EmbeddingManager as EM

class BaseGenerator(ABC):
    """
    Abstract base class for dataset generators.

    This class provides a common interface for generating data.
    It is intended to be subclassed by specific generators for different densities.

    Methods:
        generate: Generate points.
        make_random_coating: Generate coating with random materials and thicknesses.
        make_point: Generate a single point. Must be implemented by subclasses.
    """

    def __init__(self, num_points: int = 1, batch_size: int = 256):
        """
        Initialise a BaseGenerator instance.

        Args:
            num_points: Number of points to generate. Defaults to 1.
        """
        self.num_points = num_points
        self.batch_size = min(num_points, batch_size)
        self.TOLERANCE = CM().get('tolerance')

    def make_materials_mask(self, num_points: int, num_thin_films: int):
        # create mask
        # random number of thin films per point
        ones_count = torch.randint(low=CM().get('layers.min'), high=CM().get('layers.max') + 1, size=(num_points,),
                                   device=CM().get('device'))
        # index grid for columns
        col_idx = torch.arange(num_thin_films, device=CM().get('device')).unsqueeze(0)
        # compare with ones count
        materials_mask = col_idx < ones_count.unsqueeze(1)
        return materials_mask.int()

    def make_air_mask(self, num_points: int, num_thin_films: int):
        base_mask = self.make_materials_mask(num_points, num_thin_films)
        # extend mask by 2: substrate at the start and (final) air film at the end
        extended_mask = torch.cat([torch.ones((num_points, 1), device = CM().get('device')), base_mask, torch.zeros((num_points, 1), device = CM().get('device'))], dim=1)
        # invert mask for air
        return (~(extended_mask.bool()))

    def make_thicknesses(self, num_points: int):
        num_thin_films = CM().get('num_layers')
        thicknesses = torch.rand((num_points, num_thin_films), device = CM().get('device')) * 0.2
        materials_mask = self.make_materials_mask(num_points, num_thin_films)

        # append substrate with thickness 1
        substrate_thicknesses = torch.ones((num_points, 1), device = CM().get('device'))
        thicknesses = torch.cat([substrate_thicknesses, thicknesses], dim=1)

        # append air
        air_thicknesses = torch.zeros((num_points, num_thin_films + 2), device = CM().get('device'))
        extended_mask = self.make_air_mask(num_points, num_thin_films)
        # add air to thicknesses: first air film has thickness 1, the rest 0
        first_true_idx = extended_mask.int().argmax(dim=1)
        air_thicknesses_mask = torch.zeros_like(extended_mask, dtype=torch.bool, device = CM().get('device'))
        air_thicknesses_mask[torch.arange(extended_mask.size(0)), first_true_idx] = extended_mask[
            torch.arange(extended_mask.size(0)), first_true_idx]  # only if there was a True
        air_thicknesses[air_thicknesses_mask] = 1
        # extend thicknesses to add (final) air film
        thicknesses = torch.cat([thicknesses, torch.zeros((num_points, 1), device = CM().get('device'))], dim=1)
        thicknesses[extended_mask] = air_thicknesses[extended_mask]
        return thicknesses

    def make_materials_choice(self, num_points: int):
        num_thin_films = CM().get('num_layers')
        num_materials = len(CM().get('materials.thin_films'))
        # indices of thin film materials
        # index 0 and 1 reserved for substrate and air
        # as soon as torch supports random choice, change this to torch
        materials_choice = np.random.choice(num_materials, size=(num_points, num_thin_films), replace=True) + 2
        materials_choice = torch.from_numpy(materials_choice).int()
        materials_choice = materials_choice.to(CM().get('device'))

        # mask array
        materials_mask = self.make_materials_mask(num_points, num_thin_films)
        materials_choice = materials_choice * materials_mask

        # append substrate with index 0
        substrate = torch.zeros((num_points, 1), device = CM().get('device'))
        materials_choice = torch.cat([substrate, materials_choice], dim=1)

        # append air with index 1
        air = torch.ones((num_points, num_thin_films + 2), device = CM().get('device'))
        extended_mask = self.make_air_mask(num_points, num_thin_films)
        # extend materials choice to add (final) air film
        materials_choice = torch.cat([materials_choice, torch.ones((num_points, 1), device = CM().get('device'))], dim=1)
        materials_choice[extended_mask] = air[extended_mask]
        return materials_choice.int()

    def get_materials_embeddings(self, materials_indices: torch.Tensor):
        embeddings = EM().get_embeddings_lookup()
        return embeddings[materials_indices]

    @abstractmethod
    def make_points(self, num_points: int):
        """Generate points. Must be implemented by subclasses."""
        pass

    def generate(self):
        points = []
        for batch in range(math.ceil(self.num_points / self.batch_size)):
            num_points = min(self.batch_size, self.num_points - batch * self.batch_size)
            gen = self.make_points(num_points)
            points.append(gen)
        return points