import torch
from BaseGenerator import BaseGenerator
import random
import sys
sys.path.append(sys.path[0] + '/../..')
from data.values.ReflectivePropsPattern import ReflectivePropsPattern
from utils.ConfigManager import ConfigManager as CM

class ExplicitPropsGenerator(BaseGenerator):
    def __init__(self, num_points):
        super().__init__(num_points)
        self.MIN_NUM_REGIONS = 1
        self.MAX_NUM_REGIONS = 7

    def generate_random_value(self):
        if random.random() < 0.5:
            return random.choice([0, 1])
        else:
            return random.random()

    def make_point(self):
        num_regions = random.randint(self.MIN_NUM_REGIONS, self.MAX_NUM_REGIONS)
        region_indices = sorted(random.sample(range(CM().get('wavelengths').size()[0]), num_regions * 2))
        values = [self.generate_random_value() for _ in range(num_regions)]

        lower_bound = torch.zeros((1, CM().get('wavelengths').size()[0]), device = CM().get('device'))
        upper_bound = torch.ones((1, CM().get('wavelengths').size()[0]), device = CM().get('device'))

        for i in range(num_regions):
            lower_bound[:, region_indices[i * 2]:region_indices[i * 2 + 1]] = values[i]
            upper_bound[:, region_indices[i * 2]:region_indices[i * 2 + 1]] = values[i]

        lower_bound = torch.clamp(lower_bound - self.TOLERANCE / 2, 0, 1)
        upper_bound = torch.clamp(upper_bound + self.TOLERANCE / 2, 0, 1)

        result = ReflectivePropsPattern(lower_bound, upper_bound)

        return result