from abc import ABC, abstractmethod
import random
from typing import Union

import torch
import numpy as np
import math
import sys

sys.path.append(sys.path[0] + '/../..')
from data.values.Coating import Coating
from utils.ConfigManager import ConfigManager as CM

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

    def __init__(self, num_points: int = 1, batch_size: int = 1024):
        """
        Initialise a BaseGenerator instance.

        Args:
            num_points: Number of points to generate. Defaults to 1.
        """
        self.num_points = num_points
        self.batch_size = min(num_points, batch_size)
        self.TOLERANCE = CM().get('tolerance')

    def make_air_pad_mask(self, num_points: int, num_thin_films: int):
        # create mask
        # random number of thin films per point
        ones_count = torch.randint(low=CM().get('layers.min'), high=CM().get('layers.max') + 1, size=(num_points,),
                                   device=CM().get('device'))
        # index grid for columns
        col_idx = torch.arange(num_thin_films, device=CM().get('device')).unsqueeze(0)
        # compare with ones count: columns with index greater or equal to ones_count should be air
        materials_mask = col_idx >= ones_count.unsqueeze(1)
        return materials_mask

    def make_thicknesses(self, materials_choice: torch.Tensor):
        assert len(materials_choice.shape) == 2, "Materials choice must be of shape num_points x num_layers"
        num_points, num_thin_films = materials_choice.shape
        thicknesses = torch.rand((num_points, num_thin_films), device = CM().get('device')) * CM().get('thicknesses_max')
        substrate_mask = materials_choice == 0
        air_mask = materials_choice == 1

        mean = CM().get('thicknesses_max') / 2

        # set substrate to thickness mean
        thicknesses[substrate_mask] = mean

        # set air to thickness mean
        thicknesses[air_mask] = mean

        return thicknesses

    def sample_with_delayed_replacement(self, a: int,  size: Union[int, list[int]]):
        assert isinstance(size, int) or ((isinstance(size, list) or isinstance(size, tuple)) and len(size) == 2), f"size must be of type integer, list, or tuple. Found {type(size)}"
        if isinstance(size, int):
            batch_size = 1
            seq_len = size
        else:
            batch_size, seq_len = size
        rng = np.random.default_rng()

        choice_for_first = rng.choice(a, size = (batch_size, 1), replace = True)
        choice_for_subsqnt = rng.choice(a - 1, size = (batch_size, seq_len - 1), replace = True) + 1

        first_choice = np.concat([choice_for_first, choice_for_subsqnt], axis = -1)
        cumulated = np.cumsum(first_choice, axis = -1)
        return cumulated % a

    def make_materials_choice(self, num_points: int):
        # number of layers per coating
        num_thin_films = CM().get('num_layers')
        # number of available materials
        num_materials = len(CM().get('materials.thin_films'))
        # index 0 and 1 reserved for substrate and air respectively
        # as soon as torch supports random choice, change this to torch
        materials_choice = self.sample_with_delayed_replacement(num_materials, size = (num_points, num_thin_films)) + 2
        # materials_choice = np.random.choice(num_materials, size=(num_points, num_thin_films), replace=True) + 2
        materials_choice = torch.from_numpy(materials_choice).int()
        materials_choice = materials_choice.to(CM().get('device'))

        # if air_pad is enabled: mask array
        if CM().get('air_pad'):
            air_pad_mask = self.make_air_pad_mask(num_points, num_thin_films)
            materials_choice[air_pad_mask] = 1

        # append air with index 1
        air = torch.ones((num_points, 1), device = CM().get('device'), dtype = torch.int)
        materials_choice = torch.cat([materials_choice, air], dim = 1)

        # append substrate with index 0
        substrate = torch.zeros((num_points, 1), device = CM().get('device'), dtype = torch.int)
        materials_choice = torch.cat([substrate, materials_choice], dim = 1)

        return materials_choice.int()


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