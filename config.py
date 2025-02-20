import torch
import numpy as np
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

num_stacks = 1

materials_file = "Mirror Design with AI/Simple Designs/Filters/Design narrow Bandpass - optimized.dsg"

start_wl = 300e-9
end_wl = 1100e-9
steps = 1000
wavelengths = torch.linspace(start_wl, end_wl, steps, device = device)

theta = torch.tensor(np.linspace(0, 0, 1) * (np.pi / 180), dtype = torch.float32).to(device)

tolerance = 0.1 #default: 1.0e-3

learning_rate = 0.001
num_epochs = 100
