import torch
import sys
sys.path.append(sys.path[0] + '/..')
from data.values.ReflectiveProps import ReflectiveProps
from config import device


def compute_loss(input: ReflectiveProps, target: ReflectiveProps):
    refs = target.get_properties()
    preds = input.get_properties()

    lower_bound, upper_bound = torch.chunk(refs, 2, dim=1)
    lower_bound = lower_bound.reshape(target.get_steps()).to(device)
    upper_bound = upper_bound.reshape(target.get_steps()).to(device)

    upper_error = torch.clamp(preds - upper_bound, 0, 1)
    lower_error = torch.clamp(lower_bound - preds, 0, 1)

    total_error = torch.sum(upper_error ** 2 + lower_error ** 2)

    return total_error
