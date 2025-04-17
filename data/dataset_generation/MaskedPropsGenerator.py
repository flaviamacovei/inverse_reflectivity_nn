import random
import torch
import sys
sys.path.append(sys.path[0] + '/../..')
from data.values.ReflectivePropsPattern import ReflectivePropsPattern
from data.values.Coating import Coating
from forward.forward_tmm import coating_to_reflective_props
from utils.ConfigManager import ConfigManager as CM
from data.dataset_generation.BaseGenerator import BaseGenerator

class MaskedPropsGenerator(BaseGenerator):
    """
    Masked Properties Generator class for generating datasets of density 'masked'.

    A 'masked' point contains reflectivity information at random intervals of wavelengths.
    Unmasked interval: reflectivity information available at every wavelength, lower and upper bound differ by no more than config.tolerance.
    Masked interval: no reflectivity information, lower bound = 0 and upper bound = 1
    See readme for more information and examples.

    Methods:
        make_point: Generate a 'masked' point.
    """
    def __init__(self, num_points = 1):
        """Initialise a MaskedPropsGenerator instance."""
        super().__init__(num_points)
        self.MIN_NUM_MASKS = 1
        self.MAX_NUM_MASKS = 7

    def make_point(self):
        """
        Generate a 'masked' point.

        Returns:
            pattern: ReflectivePropsPattern instance with masked and unmasked intervals.
            coating: corresponding Coating instance.
        """
        num_masks = random.randint(self.MIN_NUM_MASKS, self.MAX_NUM_MASKS)
        # intervals of masking are determined by randomly sampling indices and in sorted order, selecting pairs
        # this ensures that the masked intervals are non-overlapping
        mask_indices = sorted(random.sample(range(CM().get('wavelengths').shape[0]), num_masks * 2))

        coating = self.make_random_coating()
        reflective_props_tensor = coating_to_reflective_props(coating).get_value()

        lower_bound = reflective_props_tensor - self.TOLERANCE / 2
        upper_bound = reflective_props_tensor + self.TOLERANCE / 2

        mask = torch.zeros(CM().get('wavelengths').size()[0], device = CM().get('device'))
        # set mask to 1 in every masked interval
        for i in range(num_masks):
            mask[mask_indices[i * 2]:mask_indices[i * 2 + 1]] = 1

        lower_bound = torch.clamp(lower_bound - mask, 0, 1)
        upper_bound = torch.clamp(upper_bound + mask, 0, 1)

        pattern = ReflectivePropsPattern(lower_bound, upper_bound)

        return pattern.to('cpu'), coating.to('cpu')
