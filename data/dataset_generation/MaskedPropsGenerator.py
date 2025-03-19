from BaseGenerator import BaseGenerator
import random
import torch
import sys
sys.path.append(sys.path[0] + '/../..')
from data.values.ReflectivePropsPattern import ReflectivePropsPattern
from data.values.Coating import Coating
from forward.forward_tmm import coating_to_reflective_props
from utils.ConfigManager import ConfigManager as CM

class MaskedPropsGenerator(BaseGenerator):
    def __init__(self, num_points):
        super().__init__(num_points)
        self.MIN_NUM_MASKS = 1
        self.MAX_NUM_MASKS = 7

    def make_point(self):
        num_masks = random.randint(self.MIN_NUM_MASKS, self.MAX_NUM_MASKS)
        mask_indices = sorted(random.sample(range(CM().get('wavelengths').size()[0]), num_masks * 2))

        coating = self.make_random_coating()
        reflective_props_tensor = coating_to_reflective_props(coating).get_value()

        lower_bound = reflective_props_tensor - self.TOLERANCE / 2
        upper_bound = reflective_props_tensor + self.TOLERANCE / 2

        mask = torch.zeros(CM().get('wavelengths').size()[0], device = CM().get('device'))
        for i in range(num_masks):
            mask[mask_indices[i * 2]:mask_indices[i * 2 + 1]] = 1

        lower_bound = torch.clamp(lower_bound - mask, 0, 1)
        upper_bound = torch.clamp(upper_bound + mask, 0, 1)

        pattern = ReflectivePropsPattern(lower_bound, upper_bound)

        return pattern, coating
