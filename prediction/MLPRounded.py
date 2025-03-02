import torch
from torch.utils.data import DataLoader
import sys
sys.path.append(sys.path[0] + '/..')
from prediction.BasePredictionEngine import BasePredictionEngine
from prediction.relaxation.RelaxedFeedForward import RelaxedFeedForward
from prediction.discretisation.Rounder import Rounder
from utils.ConfigManager import ConfigManager as CM
from data.dataloaders.DynamicDataloader import DynamicDataloader


class MLPRounded(BasePredictionEngine):
    def __init__(self, num_layers):
        dataloader = DynamicDataloader(batch_size=CM().get('training.batch_size'), shuffle=True)
        dataloader.load_data(CM().get('dataset_files'))
        self.relaxed_solver = RelaxedFeedForward(dataloader, num_layers)
        self.discretiser = Rounder(self.relaxed_solver)

    def train_relaxed_engine(self):
        print("Training relaxed engine...")
        self.relaxed_solver.train()
        print("Training complete")
