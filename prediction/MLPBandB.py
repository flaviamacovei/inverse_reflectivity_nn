import torch
from torch.utils.data import DataLoader
import sys
sys.path.append(sys.path[0] + '/..')
from prediction.BasePredictionEngine import BasePredictionEngine
from prediction.relaxation.RelaxedFeedForward import RelaxedFeedForward
from prediction.discretisation.BranchAndBound import BranchAndBound
from config import batch_size, dataset_files, device, thicknesses_bounds, refractive_indices_bounds, num_epochs
from data.dataloaders.DynamicDataloader import DynamicDataloader


class MLPBandB(BasePredictionEngine):
    def __init__(self, num_layers):
        switch_condition = lambda epoch: epoch % max(1, num_epochs // 3) == 0
        dataloader = DynamicDataloader(batch_size = batch_size, shuffle = False, switch_condition = switch_condition)
        dataloader.load_data(dataset_files)
        self.relaxed_solver = RelaxedFeedForward(dataloader, num_layers)
        self.discretiser = BranchAndBound(self.relaxed_solver)

    def train_relaxed_engine(self):
        print("Training relaxed engine...")
        self.relaxed_solver.train()
        print("Training complete")

    def load_relaxed_engine(self, filepath: str):
        self.relaxed_solver = torch.load(filepath)
        print("Relaxed engine loaded")
