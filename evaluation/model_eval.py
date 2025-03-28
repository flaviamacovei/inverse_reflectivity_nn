from torch.utils.data import DataLoader
import torch
import wandb
import sys
sys.path.append(sys.path[0] + '/..')
from prediction.BaseModel import BaseModel
from data.values.ReflectivePropsPattern import ReflectivePropsPattern
from forward.forward_tmm import coating_to_reflective_props
from evaluation.loss import match
from utils.ConfigManager import ConfigManager as CM
from ui.visualise import visualise

def evaluate_model(model: BaseModel):
    print("Evaluating model...")
    batch_size = 10
    densities = ["complete", "masked", "explicit"]
    total_error = 0
    for density in densities:
        dataloader = init_dataloader(density, batch_size)
        error = evaluate_per_density(model, dataloader, save_visualisation = False)
        total_error += error
        if CM().get('wandb.log'):
            wandb.log({f"{density}_error": error})
        print(f"{density} error: {error}")
    if CM().get('wandb.log'):
        wandb.log({"total_error": total_error})
    print(f"total error: {total_error}")
    print("Evaluation complete.")

def init_dataloader(density: str, batch_size: int):
    dataset = torch.load(f"data/datasets/validation/free_{density}_100.pt")
    return DataLoader(dataset, batch_size = batch_size, shuffle = False)

def evaluate_per_density(model: BaseModel, dataloader: DataLoader, save_visualisation = False):
    error = 0
    for i, batch in enumerate(dataloader):
        features = batch[0].float().to(CM().get('device'))
        lower_bound, upper_bound = features.chunk(2, dim=1)
        pattern = ReflectivePropsPattern(lower_bound, upper_bound)
        coating = model.predict(pattern)
        preds = coating_to_reflective_props(coating)
        if save_visualisation:
            visualise(refs = pattern, preds = preds, filename = "evaluation_{i}")
        error += match(preds, pattern).item()
    return error

def test_model(model: BaseModel):
    print("Testing model...")
    batch_size = 10
    dataset = torch.load(f"data/datasets/test_data/test_data.pt")
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False)
    test_error = evaluate_per_density(model, dataloader, save_visualisation = True)
    print(f"test error: {test_error}")