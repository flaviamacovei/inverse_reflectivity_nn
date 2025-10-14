import torch
import sys
sys.path.append(sys.path[0] + '/..')
from data.values.ReflectivityPattern import ReflectivityPattern
from data.values.ReflectivityValue import ReflectivityValue


def match(input: ReflectivityValue, target: ReflectivityPattern, reduction: str = 'mean'):
    """
    Match a reflectivity value to a reflectivity pattern.

    Args:
        input: reflectivity value object.
        target: reflectivity pattern object.
    """
    lower_bound = target.get_lower_bound()
    upper_bound = target.get_upper_bound()
    value = input.get_value()

    masked = lower_bound.eq(0) & upper_bound.eq(1)
    masked_factor = masked.sum(dim = -1, keepdim = True)
    batch_size, wl_size = value.shape
    scale = batch_size * (wl_size - masked_factor)
    upper_error = value - upper_bound
    upper_error = upper_error.clamp(min = 0)
    lower_error = lower_bound - value
    lower_error = lower_error.clamp(min = 0)

    total_error = (upper_error ** 2 + lower_error ** 2) / scale

    if reduction == 'none':
        return total_error
    elif reduction == 'sum':
        return total_error.sum()
    else:
        return total_error.mean()
