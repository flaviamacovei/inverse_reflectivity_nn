from torch.utils.data import DataLoader
import torch
import wandb
import sys
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
    print("Evaluating model...")
    densities = ["complete", "masked", "explicit"]
    density_errors = []
    for density in densities:
        try:
            dataloader = init_dataloader(density)
        except FileNotFoundError:
            raise FileNotFoundError("Dataset in current configuration not found. Please run generate_dataset.py first.")
            return
        error = evaluate_per_density(model, dataloader, save_visualisation = False)
        try:
            if CM().get('wandb.log'):
                wandb.log({f"{density}_error": error})
        except:
            pass
        print(f"{density} error: {error}")
        density_errors.append(error)
    total_error = sum(density_errors)
    try:
        if CM().get('wandb.log'):
            wandb.log({"total_error": total_error})
    except:
        pass
    print(f"total error: {total_error}")
    print("Evaluation complete.")
    return density_errors

def init_dataloader(density: str):
    """
    Initialise validation dataloader for specified density.

    Args:
        density: Density for which to load dataset.
    """
    batch_size = 10
    filename = get_dataset_name("validation", density)
    try:
        dataset = torch.load(filename, weights_only = False)
    except FileNotFoundError:
        raise FileNotFoundError("Dataset in current configuration not found. Please run generate_dataset.py first.")
    except AttributeError:
        raise FileNotFoundError("Dataset in current configuration not found. Please run generate_dataset.py first.")
    return DataLoader(dataset, batch_size = batch_size, shuffle = False)

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
    print("Testing model...")
    batch_size = 1
    dataset = torch.load(f"data/datasets/test_data/test_data.pt", weights_only = False)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False)
    test_error = evaluate_per_density(model, dataloader, save_visualisation = True)
    try:
        if CM().get('wandb.log'):
            wandb.log({"test_error": test_error})
    except:
        pass
    print(f"test error: {test_error}")
    return test_error