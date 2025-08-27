import random
import torch
import numpy as np
import sys
sys.path.append(sys.path[0] + '/../..')
from data.values.ReflectivityPattern import ReflectivityPattern
from data.values.Coating import Coating
from forward.forward_tmm import coating_to_reflectivity
from utils.ConfigManager import ConfigManager as CM
from data.dataset_generation.BaseGenerator import BaseGenerator

class MaskedPatternGenerator(BaseGenerator):
    """
    Masked Reflectivity Pattern Generator class for generating datasets of density 'masked'.

    A 'masked' point contains reflectivity information at random intervals of wavelengths.
    Unmasked interval: reflectivity information available at every wavelength, lower and upper bound differ by no more than config.tolerance.
    Masked interval: no reflectivity information, lower bound = 0 and upper bound = 1
    See readme for more information and examples.

    Methods:
        make_point: Generate a 'masked' point.
    """
    def __init__(self, num_points = 1, batch_size: int = 256):
        """Initialise a MaskedPatternGenerator instance."""
        super().__init__(num_points, batch_size)
        self.MIN_NUM_MASKS = 1
        self.MAX_NUM_MASKS = 7

    def make_mask(self, num_points: int):
        length = CM().get('wavelengths').shape[0]
        # random number of marks for each point
        num_marks = torch.randint(self.MIN_NUM_MASKS, self.MAX_NUM_MASKS + 1, (num_points, 1), device = CM().get('device'))
        # set the first num_marks elements of marks to 1, the rest to 0
        marks = torch.linspace(0, length - 1, length, device = CM().get('device'))[None, :].repeat(num_points, 1)
        marks = (marks <= num_marks)
        # shuffle positions along row dimension
        shuffle_indices = torch.argsort(torch.rand(*marks.shape), dim=1)
        marks = marks[torch.arange(marks.shape[0], device = CM().get('device')).unsqueeze(-1), shuffle_indices]
        marks = marks.float()

        # 1d convolution
        # ensure odd kernel size
        kernel_size = 2 * (length // 16) + 1
        kernel = torch.ones((kernel_size,), device = CM().get('device')) / kernel_size

        blurred = torch.nn.functional.conv1d(marks[:, None, :], kernel[None, None, :], padding='same').squeeze()

        # cut off at threshold
        threshold = kernel[0]
        mask = (~torch.isclose(blurred, threshold)).int()
        return mask

    def make_points(self, num_points: int):
        """
        Generate 'masked' points.

        Returns:
            pattern: ReflectivityPattern instance with masked and unmasked intervals.
            coating: corresponding Coating instance.
        """
        materials_indices = self.make_materials_choice(num_points)
        thicknesses = self.make_thicknesses(materials_indices)

        # make features
        embedding = self.get_materials_embeddings(materials_indices)
        coating_encoding = torch.cat([thicknesses[:, :, None], embedding], dim=2).float()
        coating = Coating(coating_encoding)

        mask = self.make_mask(num_points)

        reflectivity = coating_to_reflectivity(coating).get_value().float()

        lower_bound = reflectivity - self.TOLERANCE / 2
        upper_bound = reflectivity + self.TOLERANCE / 2

        lower_bound = torch.clamp(lower_bound - mask, 0, 1)
        upper_bound = torch.clamp(upper_bound + mask, 0, 1)

        pattern = ReflectivityPattern(lower_bound, upper_bound)

        return pattern, coating