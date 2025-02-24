import torch
from torch.utils.data import DataLoader
import sys
sys.path.append(sys.path[0] + '/..')
from prediction.BasePredictionEngine import BasePredictionEngine
from prediction.relaxation.RelaxedFeedForward import RelaxedFeedForward
from prediction.discretisation.Rounder import Rounder
from config import batch_size, dataset_file, device, thicknesses_bounds, refractive_indeces_bounds


class FFRounded(BasePredictionEngine):
    def __init__(self, num_layers):
        dataset = torch.load("data/datasets/" + dataset_file)
        dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False)
        thicknesses_lower_bound = torch.ones((1, num_layers), device = device) * thicknesses_bounds[0]
        thicknesses_upper_bound = torch.ones((1, num_layers), device = device) * thicknesses_bounds[1]
        refractive_index_lower_bound = torch.ones((1, num_layers), device = device) * refractive_indeces_bounds[0]
        refractive_index_upper_bound = torch.ones((1, num_layers), device = device) * refractive_indeces_bounds[1]
        self.relaxed_solver = RelaxedFeedForward(dataloader, num_layers, (thicknesses_lower_bound, thicknesses_upper_bound), (refractive_index_lower_bound, refractive_index_upper_bound))
        self.discretiser = Rounder(self.relaxed_solver)

    def train_relaxed_engine(self):
        print("Training relaxed engine...")
        self.relaxed_solver.train()
        print("Training complete")
