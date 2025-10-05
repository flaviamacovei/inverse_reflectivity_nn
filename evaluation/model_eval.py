from torch.utils.data import DataLoader
import torch
import wandb
import sys

from data.dataloaders.DynamicDataloader import DynamicDataloader

sys.path.append(sys.path[0] + '/..')
from prediction.BaseModel import BaseModel
from data.values.ReflectivityPattern import ReflectivityPattern
from forward.forward_tmm import coating_to_reflectivity
from evaluation.loss import match
from utils.ConfigManager import ConfigManager as CM
from utils.data_utils import get_dataset_name
from ui.visualise import visualise

def evaluate_model(model: BaseModel):
    """
    Evaluate model on validation dataset.

    Args:
        model: Prediction model to evaluate.
    """
    densities = ["complete", "masked", "explicit"]
    density_errors = dict()
    for density in densities:
        try:
            dataloader = init_dataloader(density)
        except FileNotFoundError:
            raise FileNotFoundError("Dataset in current configuration not found. Please run generate_dataset.py first.")
        error = evaluate_per_density(model, dataloader, save_visualisation = False)
        density_errors[density] = error
    return density_errors

def init_dataloader(density: str):
    """
    Initialise validation dataloader for specified density.

    Args:
        density: Density for which to load dataset.
    """
    batch_size = 10
    filename = get_dataset_name("validation", density)
    if filename is None:
        raise FileNotFoundError("Dataset in current configuration not found. Please run generate_dataset.py first.")
    dataloader = DynamicDataloader(batch_size, shuffle = False)
    dataloader.load_val(density)
    return dataloader

def evaluate_per_density(model: BaseModel, dataloader: DataLoader, save_visualisation = False):
    """
    Evaluate model for one density.

    Args:
        model: Prediction model to evaluate.
        dataloader: Evaluation dataloader.
        save_visualisation: Whether to save visualisation of predictions.
    """
    error = 0
    for i, batch in enumerate(dataloader):
        features = batch[0].float().to(CM().get('device'))
        lower_bound, upper_bound = features.chunk(2, dim=1)
        pattern = ReflectivityPattern(lower_bound, upper_bound)
        coating = model.predict(pattern)
        preds = coating_to_reflectivity(coating)
        if save_visualisation:
            visualise(refs = pattern, preds = preds, filename = f"evaluation_{i}")
        # evaluation uses free loss
        error += match(preds, pattern).item() / len(dataloader)
    return error

def test_model(model: BaseModel):
    """
    Evaluate model on test dataset.

    Args:
        model: Prediction model to evaluate.
    """
    batch_size = 1
    test_data_path = f"data/datasets/test_data/test_data.pt"
    dataloader = DynamicDataloader(batch_size, False)
    dataloader.load_test(test_data_path)
    test_error = evaluate_per_density(model, dataloader, save_visualisation = True)
    return test_error