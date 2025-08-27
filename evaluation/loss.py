import torch
import sys
sys.path.append(sys.path[0] + '/..')
from data.values.ReflectivityPattern import ReflectivityPattern
from data.values.ReflectivityValue import ReflectivityValue


def match(input: ReflectivityValue, target: ReflectivityPattern):
    """
    Match a reflectivity value to a reflectivity pattern.

    Args:
        input: reflectivity value object.
        target: reflectivity pattern object.
    """
    batch_size = input.get_value().shape[0]
    wl_size = input.get_value().shape[1]
    scale_mean = batch_size * wl_size
    upper_error = input.get_value() - target.get_upper_bound()
    upper_error = upper_error.clamp(min = 0)
    lower_error = target.get_lower_bound() - input.get_value()
    lower_error = lower_error.clamp(min = 0)

    total_error = torch.sum(upper_error ** 2 + lower_error ** 2) / scale_mean

    return total_error
