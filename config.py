import torch
import numpy as np
import random

device = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'

num_stacks = 1

materials_file = "Mirror Design with AI/Simple Designs/Filters/Design narrow Bandpass - optimized.dsg"

start_wl = 500e-9
end_wl = 1500e-9
steps = 10
wavelengths = torch.linspace(start_wl, end_wl, steps)

theta = torch.tensor(np.linspace(0, 0, 1) * (np.pi / 180), dtype = torch.float32).to(device)

tolerance = 0 #default: 1.0e-3
