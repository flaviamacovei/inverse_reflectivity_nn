import pandas as pd
import torch
import sys


sys.path.append(sys.path[0] + '/..')
from utils.data_utils import load_config
from utils.ConfigManager import ConfigManager as CM
from data.dataloaders.DynamicDataloader import DynamicDataloader
from data.values.ReflectivePropsPattern import ReflectivePropsPattern
from data.values.ReflectivePropsValue import ReflectivePropsValue
from utils.data_utils import get_dataset_name
from prediction.BaseModel import BaseModel
from prediction.GradientModel import GradientModel
from prediction.MLP import MLP
from prediction.Transformer import Transformer
from forward.forward_tmm import coating_to_reflective_props
from ui.visualise import visualise

def load_pattern():
    lower_bound = None
    upper_bound = None
    for density in ['complete', 'masked', 'explicit']:
        dataloader = DynamicDataloader(CM().get('training.dataset_size'), shuffle=False)
        filepath = get_dataset_name("training", density)
        dataloader.load_data(filepath, weights_only=False)
        dataset = dataloader.dataset
        local_lower_bound, local_upper_bound = torch.chunk(dataset[:][0], 2, dim=-1)
        local_lower_bound = local_lower_bound.to(CM().get('device'))
        local_upper_bound = local_upper_bound.to(CM().get('device'))
        lower_bound = local_lower_bound if lower_bound == None else torch.cat([lower_bound, local_lower_bound], dim=0)
        upper_bound = local_upper_bound if upper_bound == None else torch.cat([upper_bound, local_upper_bound], dim=0)
    return ReflectivePropsPattern(lower_bound=lower_bound, upper_bound=upper_bound)

def random_baseline():
    pattern = load_pattern()
    error = pairwise_distance(pattern)

    return evaluate_error(error)


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
    print(error.shape)
    smallest_idx = torch.argmin(error)
    multiindex = torch.unravel_index(smallest_idx, error.shape)

    p1 = ReflectivePropsValue(p1_lower[multiindex[0]], p1_upper[multiindex[0]])
    p2 = ReflectivePropsValue(p2_lower[multiindex[1]], p2_upper[multiindex[1]])
    visualise(refs = p1, filename = "p1")
    visualise(refs = p2, filename = "p2")
    print(multiindex)
    smallest = error[multiindex[0], multiindex[1]]
    print(smallest)
    indices = torch.triu_indices(num, num, offset = 1)
    return error[indices[0], indices[1]]

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

    total_error = torch.sum(upper_error ** 2 + lower_error ** 2, dim = -1)

    return total_error

def evaluate_model(model: BaseModel):
    target = load_pattern()
    coating = model.predict(target)
    preds = coating_to_reflective_props(coating)
    error = match(preds, target)
    return evaluate_error(error)

def evaluate_error(error):
    min = error.min().item()
    max = error.max().item()
    mean = error.mean().item()
    median = error.median().item()
    q_25 = error.quantile(0.25).item()
    q_75 = error.quantile(0.75).item()
    return {'min': min, 'q_25': q_25, 'mean': mean, 'median': median, 'q_75': q_75, 'max': max}


def evaluate_all_models():
    results = {
        'model': [],
        'min': [],
        'q_25': [],
        'mean': [],
        'median': [],
        'q_75': [],
        'max': []
    }
    random = random_baseline()
    for key in results.keys():
        if key == 'model':
            results[key].append('random')
        else:
            results[key].append(random[key])
    model_classes = {
        'gradient': GradientModel,
        # 'mlp': MLP,
        # 'transformer': Transformer
    }
    # for type in ['gradient']:#, 'mlp', 'transformer']:
    #     ModelClass = model_classes[type]
    #     model = ModelClass()
    #     evaluation = evaluate_model(model)
    #     for key in results.keys():
    #         if key == 'model':
    #             results[key].append(type)
    #         else:
    #             results[key].append(evaluation[key])

    results = pd.DataFrame(results)
    return results

    

if __name__ == '__main__':
    results = evaluate_all_models()
    print(results)