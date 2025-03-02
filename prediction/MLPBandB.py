import torch
from torch.utils.data import DataLoader
import sys
sys.path.append(sys.path[0] + '/..')
from prediction.BasePredictionEngine import BasePredictionEngine
from prediction.relaxation.RelaxedFeedForward import RelaxedFeedForward
from prediction.discretisation.BranchAndBound import BranchAndBound
from utils.ConfigManager import ConfigManager as CM
from data.dataloaders.DynamicDataloader import DynamicDataloader


class MLPBandB(BasePredictionEngine):
    def __init__(self, num_layers):
        dataloader = DynamicDataloader(batch_size = CM().get('training.batch_size'), shuffle = False)
        dataloader.load_data(CM().get('dataset_files'))
        self.relaxed_solver = RelaxedFeedForward(dataloader, num_layers)
        self.discretiser = BranchAndBound(self.relaxed_solver)

    def train_relaxed_engine(self):
        print("Training relaxed engine...")
        self.relaxed_solver.train()
        print("Training complete")

    def load_relaxed_engine(self, filepath: str):
        self.relaxed_solver = torch.load(filepath)
        print("Relaxed engine loaded")
