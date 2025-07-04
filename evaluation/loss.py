import torch
import sys
sys.path.append(sys.path[0] + '/..')
from data.values.ReflectivePropsPattern import ReflectivePropsPattern
from data.values.ReflectivePropsValue import ReflectivePropsValue


def match(input: ReflectivePropsValue, target: ReflectivePropsPattern):
    """
    Match a reflective properties value to a reflective properties pattern.

    Args:
        input: reflective properties value object.
        target: reflective properties pattern object.
    """
    upper_error = input.get_value() - target.get_upper_bound()
    upper_error = upper_error.clamp(min = 0)
    lower_error = target.get_lower_bound() - input.get_value()
    lower_error = lower_error.clamp(min = 0)

    total_error = torch.sum(upper_error ** 2 + lower_error ** 2)

    return total_error
