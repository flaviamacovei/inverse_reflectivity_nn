import torch
from torch.utils.data import DataLoader
import sys
sys.path.append(sys.path[0] + '/..')
from prediction.BasePredictionEngine import BasePredictionEngine
from prediction.relaxation.RelaxedFeedForward import RelaxedFeedForward
from prediction.discretisation.Rounder import Rounder
from config import batch_size, dataset_file

class FFRounded(BasePredictionEngine):
    def __init__(self, num_layers):
        dataset = torch.load("data/datasets/" + dataset_file)
        dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False)
        self.relaxed_solver = RelaxedFeedForward(dataloader, num_layers)
        self.discretiser = Rounder(self.relaxed_solver)

    def train_relaxed_engine(self):
        print("Training relaxed engine...")
        self.relaxed_solver.train()
        print("Training complete")
