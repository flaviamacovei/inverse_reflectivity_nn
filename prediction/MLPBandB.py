import torch
from torch.utils.data import DataLoader
import sys
sys.path.append(sys.path[0] + '/..')
from prediction.BasePredictionEngine import BasePredictionEngine
from prediction.relaxation.RelaxedFeedForward import RelaxedFeedForward
from prediction.discretisation.BranchAndBound import BranchAndBound
from config import batch_size, dataset_file, device, thicknesses_bounds, refractive_indices_bounds


class MLPBandB(BasePredictionEngine):
    def __init__(self, num_layers):
        dataset = torch.load("data/datasets/" + dataset_file)
        dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False)
        self.relaxed_solver = RelaxedFeedForward(dataloader, num_layers)
        self.discretiser = BranchAndBound(self.relaxed_solver)

    def train_relaxed_engine(self):
        print("Training relaxed engine...")
        self.relaxed_solver.train()
        print("Training complete")
