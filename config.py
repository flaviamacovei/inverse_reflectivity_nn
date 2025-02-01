import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_stacks = 1
materials_file = "Mirror Design with AI/Simple Designs/Filters/Design narrow Bandpass - optimized.dsg"