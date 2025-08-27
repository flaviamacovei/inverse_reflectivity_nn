import os
import time
import torch
from torch.utils.data import DataLoader
from prediction.GradientModel import GradientModel
from utils.ConfigManager import ConfigManager as CM
from data.values.ReflectivityPattern import ReflectivityPattern
from data.dataloaders.DynamicDataloader import DynamicDataloader
from ui.visualise import visualise
from forward.forward_tmm import coating_to_reflectivity

def gradient_predict():
    """Instantiate and train model."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # dataloader = DynamicDataloader(batch_size=1, shuffle=False)
    # dataloader.load_leg(0)
    # point = dataloader[0]
    # lower_bound, upper_bound = point[0][None].chunk(2, dim=1)
    # lower_bound = lower_bound.to(CM().get('device'))
    # upper_bound = upper_bound.to(CM().get('device'))

    model = GradientModel()

    lower_bound = (torch.linspace(0.3, 0.5, 1200, device = CM().get('device')) - 0.05).clamp(min = 0, max = 1)[None]
    upper_bound = (torch.linspace(0.5, 0.7, 1200, device = CM().get('device')) + 0.05).clamp(min = 0, max = 1)[None]

    target = ReflectivityPattern(lower_bound, upper_bound)


    optimal_coating = model.predict(target)
    print(optimal_coating)

    result = coating_to_reflectivity(optimal_coating)
    output = "out/gradient.png"
    if os.path.exists(output):
        os.remove(output)
    visualise(preds = result, refs = target, filename = "gradient")

if __name__ == "__main__":
    gradient_predict()
