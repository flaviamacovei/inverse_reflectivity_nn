import pandas as pd
import torch
import sys
import os
import wandb
import yaml

sys.path.append(sys.path[0] + '/..')
from utils.data_utils import load_config
from utils.ConfigManager import ConfigManager as CM
from data.dataloaders.DynamicDataloader import DynamicDataloader
from data.values.ReflectivityPattern import ReflectivityPattern
from data.values.ReflectivityValue import ReflectivityValue
from utils.data_utils import get_dataset_name, get_saved_model_path
from prediction.BaseModel import BaseModel
from prediction.RandomModel import RandomModel
from prediction.BaseTrainableModel import BaseTrainableModel
from prediction.GradientModel import GradientModel
from prediction.MLP import MLP
from prediction.CNN import CNN
from prediction.Hybrid import Hybrid
from prediction.Transformer import Transformer
from forward.forward_tmm import coating_to_reflectivity
from ui.visualise import visualise_errors
from ui.cl_interact import ding

def load_pattern(type_data: str):
    if type_data == "validation":
        lower_bound = None
        upper_bound = None
        for density in ['complete', 'masked', 'explicit']:
            dataloader = DynamicDataloader(100, shuffle=False)
            own_path = os.path.realpath(__file__)
            dataloader.load_val(density)
            dataset = dataloader.dataset
            local_lower_bound, local_upper_bound = torch.chunk(dataset[:][0], 2, dim=-1)
            local_lower_bound = local_lower_bound.to(CM().get('device'))
            local_upper_bound = local_upper_bound.to(CM().get('device'))
            lower_bound = local_lower_bound if lower_bound == None else torch.cat([lower_bound, local_lower_bound], dim=0)
            upper_bound = local_upper_bound if upper_bound == None else torch.cat([upper_bound, local_upper_bound], dim=0)
        lower_bound = lower_bound.to(CM().get('device'))
        upper_bound = upper_bound.to(CM().get('device'))
        return ReflectivityPattern(lower_bound, upper_bound)
    elif type_data == "test":
        ownpath = os.path.realpath(__file__)
        dataset = torch.load(os.path.join(os.path.dirname(os.path.dirname(ownpath)), "data/datasets/test_data/test_data.pt"), weights_only = False)
        batch = dataset.tensors
        reflectivity = batch[0].to(CM().get('device'))
        lower_bound, upper_bound = torch.chunk(reflectivity, 2, -1)
        return ReflectivityPattern(lower_bound, upper_bound)
    else:
        raise ValueError(f"Invalid type {type_data}")


def match(input: ReflectivityValue, target: ReflectivityPattern):
    """
    Match a reflectivity value to a reflectivity pattern.

    Args:
        input: reflectivity value object.
        target: reflectivity pattern object.
    """
    wl_size = input.get_value().shape[1]

    upper_error = input.get_value() - target.get_upper_bound()
    upper_error = upper_error.clamp(min = 0)
    lower_error = target.get_lower_bound() - input.get_value()
    lower_error = lower_error.clamp(min = 0)

    total_error = torch.sum(upper_error ** 2 + lower_error ** 2, dim = -1) / wl_size

    return total_error

def evaluate_model(model: BaseModel, type_data: str):
    target = load_pattern(type_data)
    coating = model.predict(target)
    preds = coating_to_reflectivity(coating)
    num_points = target.get_batch_size()
    error = match(preds, target) / num_points
    return evaluate_error(error)

def evaluate_error(error):
    min = error.min().item()
    max = error.max().item()
    mean = error.mean().item()
    median = error.median().item()
    q_25 = error.quantile(0.25).item()
    q_75 = error.quantile(0.75).item()
    return {'min': min, 'q_25': q_25, 'mean': mean, 'median': median, 'q_75': q_75, 'max': max}

def evaluate_all_models(type_data: str):
    results = {
        'model': [],
        'min': [],
        'q_25': [],
        'mean': [],
        'median': [],
        'q_75': [],
        'max': []
    }
    model_classes = {
        # 'random': {'class': RandomModel},
        # 'gradient': {'class': GradientModel},
        # 'mlp': {'class': MLP},
        # 'mlp+gradient': {'class': lambda: Hybrid('mlp')},
        # 'cnn': {'class': CNN},
        # 'cnn+gradient': {'class': lambda: Hybrid('cnn')},
        'transformer_no_masks': {'class': Transformer, 'attrs': {'src_mask': False, 'tgt_struct_mask': False, 'tgt_caus_mask': False}},
        'transformer_caus_mask': {'class': Transformer, 'attrs': {'src_mask': False, 'tgt_struct_mask': False, 'tgt_caus_mask': True}},
        'transformer_struct_mask': {'class': Transformer, 'attrs': {'src_mask': False, 'tgt_struct_mask': True, 'tgt_caus_mask': False}},
        'transformer_struct_caus_mask': {'class': Transformer, 'attrs': {'src_mask': False, 'tgt_struct_mask': True, 'tgt_caus_mask': True}},
        'transformer_src_mask': {'class': Transformer, 'attrs': {'src_mask': True, 'tgt_struct_mask': False, 'tgt_caus_mask': False}},
        'transformer_src_caus_mask': {'class': Transformer, 'attrs': {'src_mask': True, 'tgt_struct_mask': False, 'tgt_caus_mask': True}},
        'transformer_src_struct_mask': {'class': Transformer, 'attrs': {'src_mask': True, 'tgt_struct_mask': True, 'tgt_caus_mask': False}},
        'transformer_all_masks': {'class': Transformer, 'attrs': {'src_mask': True, 'tgt_struct_mask': True, 'tgt_caus_mask': True}},
        # 'transformer+gradient': {'class': lambda: Hybrid('transformer')},
    }
    for type in model_classes.keys():
        ModelClass = model_classes[type]['class']
        model = ModelClass()
        if isinstance(model, BaseTrainableModel):
            attrs = {'model_details': model_classes[type]['attrs']} if 'attrs' in model_classes[type].keys() else None
            model.load_or_train(attrs)
        evaluation = evaluate_model(model, type_data)
        for key in results.keys():
            if key == 'model':
                results[key].append(type)
            else:
                results[key].append(evaluation[key])
    results = pd.DataFrame(results)
    return results

def remove_files():
    with open('data/datasets/metadata.yaml', 'r') as f:
        content = yaml.safe_load(f)
        for density in ['complete', 'masked', 'explicit']:
            filepath = get_dataset_name("training", density)
            item = next((x for x in content['datasets'] if x['title'] == filepath), None)
            content['datasets'].remove(item)
            os.remove(filepath)
    with open('data/datasets/metadata.yaml', 'w') as f:
        yaml.dump(content, f, sort_keys=False)

if __name__ == '__main__':
    for type_data in ['test']:
    # for type_data in ['validation', 'test']:
        results = evaluate_all_models(type_data)
        print("-" * 50 + f"\n{type_data.upper()} DATA\n" + "-" * 50)
        with pd.option_context('display.max_rows', None,
                               'display.max_columns', None,
                               'display.precision', 3,
                               ):
            print(results)
        visualise_errors(results, f"{type_data}_errors_t_graph", log_scale = True)
        results.to_csv(f"out/{type_data}_t_errors.csv")
        ding()
