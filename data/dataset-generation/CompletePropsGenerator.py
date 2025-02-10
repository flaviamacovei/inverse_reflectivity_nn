import torch
from BaseGenerator import BaseGenerator
import sys
sys.path.append(sys.path[0] + '/../..')
from data.values.Coating import Coating
from data.values.ReflectivePropsPattern import ReflectivePropsPattern
from forward.forward_tmm import coating_to_reflective_props
from config import device

class CompletePropsGenerator(BaseGenerator):
    def __init__(self, num_points):
        super().__init__(num_points)

    def make_point(self):
        coating = self.make_random_coating()
        reflective_props_tensor = coating_to_reflective_props(coating, self.START_WL, self.END_WL, self.STEPS).get_value()

        lower_bound = torch.clamp(reflective_props_tensor - self.TOLERANCE / 2, 0, 1)
        upper_bound = torch.clamp(reflective_props_tensor + self.TOLERANCE / 2, 0, 1)

        result = ReflectivePropsPattern(self.START_WL, self.END_WL, lower_bound, upper_bound)

        return result