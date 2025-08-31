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
from utils.data_utils import get_dataset_name
from prediction.BaseModel import BaseModel
from prediction.RandomModel import RandomModel
from prediction.BaseTrainableModel import BaseTrainableModel
from prediction.GradientModel import GradientModel
from prediction.MLP import MLP
from prediction.CNN import CNN
from prediction.MLPGradient import MLPGradient
from prediction.Transformer import Transformer
from forward.forward_tmm import coating_to_reflectivity
from data.material_embedding.EmbeddingManager import EmbeddingManager as EM

def load_pattern():
    lower_bound = None
    upper_bound = None
    for density in ['complete', 'masked', 'explicit']:
        dataloader = DynamicDataloader(100, shuffle=False)
        own_path = os.path.realpath(__file__)
        filepath = os.path.join(os.path.dirname(os.path.dirname(own_path)), get_dataset_name("validation", density))
        dataloader.load_data(filepath, weights_only=False)
        dataset = dataloader.dataset
        local_lower_bound, local_upper_bound = torch.chunk(dataset[:][0], 2, dim=-1)
        local_lower_bound = local_lower_bound.to(CM().get('device'))
        local_upper_bound = local_upper_bound.to(CM().get('device'))
        lower_bound = local_lower_bound if lower_bound == None else torch.cat([lower_bound, local_lower_bound], dim=0)
        upper_bound = local_upper_bound if upper_bound == None else torch.cat([upper_bound, local_upper_bound], dim=0)
    return ReflectivityPattern(lower_bound=lower_bound, upper_bound=upper_bound)


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

def evaluate_model(model: BaseModel):
    target = load_pattern()
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

def retrieve_loaded_model(type):
    ownpath = os.path.realpath(__file__)
    models_data_file = os.path.join(os.path.dirname(os.path.dirname(ownpath)), "out/models/models_metadata.yaml")
    props_dict = {
        "architecture": type,
        "model_details": CM().get(type),
        "num_layers": CM().get('num_layers'),
        "min_wl": CM().get('wavelengths')[0].item(),
        "max_wl": CM().get('wavelengths')[-1].item(),
        "wl_step": len(CM().get('wavelengths')),
        "polarisation": CM().get('polarisation'),
        "materials_hash": EM().hash_materials(),
        "num_materials": len(CM().get('materials.thin_films')),
        "theta": CM().get('theta').item(),
        "air_pad": CM().get('air_pad'),
        "stratified_sampling": CM().get('stratified_sampling'),
        "tolerance": CM().get('tolerance'),
        "num_points": CM().get('training.dataset_size'),
        "epochs": CM().get('training.num_epochs')
    }
    if os.path.exists(models_data_file):
        with open(models_data_file, "r") as f:
            content = yaml.safe_load(f)
            # search for properties dictionary match in metadata
            for model in content["models"]:
                if model["properties"] == props_dict:
                    return os.path.join(os.path.dirname(os.path.dirname(ownpath)), model["title"])
    return None

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
    model_classes = {
        'random': RandomModel,
        'gradient': GradientModel,
        # 'mlp': MLP,
        # 'mlp+gradient': MLPGradient,
        # 'cnn': CNN,
        'transformer': Transformer,
    }
    for type in model_classes.keys():
        ModelClass = model_classes[type]
        model = ModelClass()
        if isinstance(model, BaseTrainableModel):
            model_filename = retrieve_loaded_model(type)
            if model_filename is not None:
                trainable_model = torch.load(model_filename, weights_only = False)
                model.model = trainable_model
            else:
                print(f"Saved {type} model not found. Performing training...")
                if CM().get('wandb.log'):
                    wandb.init(
                        project=CM().get('wandb.project'),
                        config=CM().get('wandb.config')
                    )
                model.train()

        evaluation = evaluate_model(model)
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
    results = evaluate_all_models()
    results.to_csv("out/config_2.csv")
    print(results)
