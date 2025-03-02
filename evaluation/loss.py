import torch
import sys
sys.path.append(sys.path[0] + '/..')
from data.values.ReflectivePropsPattern import ReflectivePropsPattern
from data.values.ReflectivePropsValue import ReflectivePropsValue


def match(input: ReflectivePropsValue, target: ReflectivePropsPattern):
    upper_error = input.get_value() - target.get_upper_bound()
    lower_error = target.get_lower_bound() - input.get_value()

    total_error = torch.sum(upper_error ** 2 + lower_error ** 2)

    return total_error
