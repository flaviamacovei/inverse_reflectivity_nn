import torch
from torch.utils.data import TensorDataset
from ui.UserInput import UserInput

patterns = []
continue_reading = True
while continue_reading:
    ui = UserInput()
    ui.read_regions()
    patterns.append(ui.to_reflectivity_pattern())
    continue_reading = input("Specify another point? (y/n): ").lower() == 'y'

feature_tensors = []
for pattern in patterns:
    feature_tensor = torch.cat([pattern.get_lower_bound(), pattern.get_upper_bound()], dim = 1).squeeze()
    feature_tensors.append(feature_tensor)

feature_tensors = torch.stack(feature_tensors)
torch.save(TensorDataset(feature_tensors), "data/datasets/test_data/test_data_new.pt")