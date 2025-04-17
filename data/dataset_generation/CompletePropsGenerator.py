import torch
import sys
sys.path.append(sys.path[0] + '/../..')
from data.dataset_generation.BaseGenerator import BaseGenerator
from data.values.Coating import Coating
from data.values.ReflectivePropsPattern import ReflectivePropsPattern
from forward.forward_tmm import coating_to_reflective_props

class CompletePropsGenerator(BaseGenerator):
    """
    Complete Properties Generator class for generating datasets of density 'complete'.

    A 'complete' point contains reflectivity information at every wavelength.
    Lower bound and upper bound differ by no more than config.tolerance.
    See readme for more information and examples.

    Methods:
        make_point: Generate a 'complete' point.
    """
    def __init__(self, num_points = 1):
        """Initialise a CompletePropsGenerator instance."""
        super().__init__(num_points)

    def make_point(self):
        """
        Generate a 'complete' point.

        Returns:
            pattern: ReflectivePropsPattern instance where lower bound and upper bound differ by no more than config.tolerance.
            coating: corresponding Coating instance.
        """
        coating = self.make_random_coating()
        reflective_props_tensor = coating_to_reflective_props(coating).get_value()

        lower_bound = torch.clamp(reflective_props_tensor - self.TOLERANCE / 2, 0, 1)
        upper_bound = torch.clamp(reflective_props_tensor + self.TOLERANCE / 2, 0, 1)

        pattern = ReflectivePropsPattern(lower_bound, upper_bound)

        return pattern.to('cpu'), coating.to('cpu')