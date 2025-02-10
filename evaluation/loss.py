import torch
import sys
sys.path.append(sys.path[0] + '/..')
from data.values.ReflectivePropsPattern import ReflectivePropsPattern
from data.values.ReflectivePropsValue import ReflectivePropsValue


def compute_loss(input: ReflectivePropsValue, target: ReflectivePropsPattern):

    upper_error = torch.clamp(input.get_value() - target.get_upper_bound(), 0, 1)
    lower_error = torch.clamp(target.get_lower_bound() - input.get_value(), 0, 1)

    total_error = torch.sum(upper_error ** 2 + lower_error ** 2)

    return total_error
