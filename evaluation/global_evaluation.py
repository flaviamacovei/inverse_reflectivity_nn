import pandas as pd
import torch
import sys
sys.path.append(sys.path[0] + '/..')
from utils.data_utils import load_config
from utils.ConfigManager import ConfigManager as CM
from data.dataloaders.DynamicDataloader import DynamicDataloader
from data.values.ReflectivePropsPattern import ReflectivePropsPattern
from utils.data_utils import get_dataset_name

def random_baseline():
    lower_bound = None
    upper_bound = None
    for density in ['complete', 'masked', 'explicit']:
        dataloader = DynamicDataloader(CM().get('training.dataset_size'), shuffle = False)
        filepath = get_dataset_name("training", density)
        dataloader.load_data(filepath, weights_only = False)
        dataset = dataloader.dataset
        local_lower_bound, local_upper_bound = torch.chunk(dataset[:][0], 2, dim = -1)
        local_lower_bound = local_lower_bound.to(CM().get('device'))
        local_upper_bound = local_upper_bound.to(CM().get('device'))
        lower_bound = local_lower_bound if lower_bound == None else torch.cat([lower_bound, local_lower_bound], dim = 0)
        upper_bound = local_upper_bound if upper_bound == None else torch.cat([upper_bound, local_upper_bound], dim = 0)

    pattern = ReflectivePropsPattern(lower_bound = lower_bound, upper_bound = upper_bound)
    error = pairwise_distance(pattern)

    min = error.min().item()
    max = error.max().item()
    mean = error.mean().item()
    median = error.median().item()
    q_25 = error.quantile(0.25).item()
    q_75 = error.quantile(0.75).item()
    return {'min': min, 'q_25': q_25, 'mean': mean, 'median': median, 'q_75': q_75, 'max': max}


def pairwise_distance(pattern: ReflectivePropsPattern):
    epsilon = CM().get('tolerance')
    num = pattern.get_batch_size()
    repetitions = [1] * (len(pattern.get_lower_bound().shape) + 1)
    repetitions[1] = num
    p1_lower = pattern.get_lower_bound()[:, None].repeat(repetitions)
    p1_upper = pattern.get_upper_bound()[:, None].repeat(repetitions)
    p2_lower = pattern.get_lower_bound()[:, None].repeat(repetitions).transpose(0, 1)
    p2_upper = pattern.get_upper_bound()[:, None].repeat(repetitions).transpose(0, 1)

    p1_lower_mask = p1_lower == 0
    p1_upper_mask = p1_upper == 1
    horizontal_mask = torch.logical_and(p1_lower_mask, p1_upper_mask)
    p2_lower_mask = p2_lower == 0
    p2_upper_mask = p2_upper == 1
    vertical_mask = torch.logical_and(p2_lower_mask, p2_upper_mask)
    mask = (~(torch.logical_or(horizontal_mask, vertical_mask))).int()

    p1_lower = p1_lower + epsilon
    p1_upper = p1_upper - epsilon
    p2_lower = p2_lower + epsilon
    p2_upper = p2_upper - epsilon

    left_error = (p1_lower - p2_upper).clamp(min = 0)
    right_error = (p2_lower - p1_upper).clamp(min = 0)
    error = (left_error + right_error)
    error = error * mask
    error = error.sum(dim = -1)
    indices = torch.triu_indices(num, num, offset = 1)
    return error[indices[0], indices[1]]

def compare_all():
    results = pd.DataFrame()
    

if __name__ == '__main__':
    results = random_baseline()
    print(results)