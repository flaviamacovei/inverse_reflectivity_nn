import torch
import random
import sys

sys.path.append(sys.path[0] + '/../..')
from data.values.ReflectivePropsPattern import ReflectivePropsPattern
from utils.ConfigManager import ConfigManager as CM
from data.dataset_generation.BaseGenerator import BaseGenerator


class ExplicitPropsGenerator(BaseGenerator):
    """
    Explicit Properties Generator class for generating datasets of density 'explicit'.

    An 'explicit' point consists of a list of regions with explicit values for each region.
    Outside of those, no reflectivity is specified.
    See readme for more information and examples.

    Methods:
        make_point: Generate an 'explicit' point.
    """

    def __init__(self, num_points = 1):
        """Initialise a MaskedPropsGenerator instance."""
        super().__init__(num_points)
        self.MIN_NUM_REGIONS = 1
        self.MAX_NUM_REGIONS = 7

    def generate_random_value(self):
        """
        Generate a random value for a region with higher probability for 0 and 1.

        A region with a value of 0 or 1 is more often encountered in the use case.
        This is reflected in the higher probability for 0 and 1 while maintaining flexibility for hitting any value.

        Returns:
            A random value between 0 and 1.
        """
        if random.random() < 0.5:
            return random.choice([0, 1])
        else:
            return random.random()

    def make_point(self):
        """
        Generate an 'explicit' point.

        Returns:
            pattern: ReflectivePropsPattern instance with regions with explicit values.
            coating: corresponding Coating instance.
        """
        num_regions = random.randint(self.MIN_NUM_REGIONS, self.MAX_NUM_REGIONS)
        # intervals of masking are determined by randomly sampling indices and in sorted order, selecting pairs
        # this ensures that the masked intervals are non-overlapping
        region_indices = sorted(random.sample(range(CM().get('wavelengths').size()[0]), num_regions * 2))
        values = [self.generate_random_value() for _ in range(num_regions)]

        # bound initialisation: lower bound is 0, upper bound is 1
        lower_bound = torch.zeros((1, CM().get('wavelengths').size()[0]), device = CM().get('device'))
        upper_bound = torch.ones((1, CM().get('wavelengths').size()[0]), device = CM().get('device'))

        # set explicit values
        for i in range(num_regions):
            lower_bound[:, region_indices[i * 2]:region_indices[i * 2 + 1]] = values[i]
            upper_bound[:, region_indices[i * 2]:region_indices[i * 2 + 1]] = values[i]

        lower_bound = torch.clamp(lower_bound - self.TOLERANCE / 2, 0, 1)
        upper_bound = torch.clamp(upper_bound + self.TOLERANCE / 2, 0, 1)

        pattern = ReflectivePropsPattern(lower_bound, upper_bound)

        return pattern.to('cpu'), None