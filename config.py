import torch
import numpy as np
import random
import wandb

random.seed(0)
torch.manual_seed(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

num_stacks = 1


# dataset_files = ['complete_props_100.pt', 'masked_props_100.pt', 'explicit_props_100.pt']
dataset_files = ["complete_with_labels_50000.pt", "masked_with_labels_50000.pt"]
min_num_layers = 3
max_num_layers = 13

start_wl = 300e-9
end_wl = 1100e-9
steps = 1000
wavelengths = torch.linspace(start_wl, end_wl, steps, device = device)
thicknesses_bounds = (1E-9, 1E-6)
refractive_indices_bounds = (0.12, 2.25)

theta = torch.tensor(np.linspace(0, 0, 1) * (np.pi / 180), dtype = torch.float32).to(device)

tolerance = 1.0e-3

learning_rate = 0.001
num_epochs = 60
batch_size = 128

wandb.init(
    # set the wandb project where this run will be logged
    project="inverse-mirrors-nn",

    # track hyperparameters and run metadata
    config={
    "learning_rate": learning_rate,
    "architecture": "MLP",
    "epochs": num_epochs,
    "batch_size": batch_size,
    }
)
