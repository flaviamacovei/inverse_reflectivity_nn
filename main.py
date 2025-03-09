import os
import time
import torch
import wandb
from forward.forward_tmm import coating_to_reflective_props
from prediction.relaxation.BaseTrainableRelaxedSolver import BaseTrainableRelaxedSolver
from prediction.relaxation.GradientSolver import GradientSolver
from prediction.relaxation.RelaxedMLP import RelaxedMLP
from prediction.relaxation.RelaxedCNN import RelaxedCNN
from prediction.discretisation.Rounder import Rounder
from prediction.discretisation.BranchAndBound import BranchAndBound
from data.dataloaders.DynamicDataloader import DynamicDataloader
from data.values.Coating import Coating
from ui.visualise import visualise
from ui.FileInput import FileInput
from data.values.RefractiveIndex import RefractiveIndex
from data.values.ReflectivePropsPattern import ReflectivePropsPattern
from utils.ConfigManager import ConfigManager as CM
from evaluation.model_eval import evaluate
from data.material_embedding.EmbeddingManager import EmbeddingManager

def notify():
    os.system("echo -ne '\007'")

def make_random_pattern():
    random_thicknesses = torch.rand((1, CM().get('num_layers') - 2), device=CM().get('device')) * 1E-6
    random_refractive_indices = (2.25 - 0.12) * torch.rand((1, CM().get('num_layers') - 2),
                                                           device=CM().get('device')) + 0.12
    random_refractive_indices_rounded = RefractiveIndex.round_tensor(random_refractive_indices)
    coating = Coating(random_thicknesses, random_refractive_indices_rounded)
    value = coating_to_reflective_props(coating)
    lower_bound = torch.clamp(value.get_value() - CM().get('tolerance') / 2, 0, 1).float()
    upper_bound = torch.clamp(value.get_value() + CM().get('tolerance') / 2, 0, 1).float()
    return ReflectivePropsPattern(lower_bound, upper_bound)

if __name__ == "__main__":
    # relaxed_solvers = {
    #     "gradient": GradientSolver,
    #     "mlp": RelaxedMLP,
    #     "cnn": RelaxedCNN
    # }
    # discretisers = {
    #     "rounder": Rounder,
    #     "b&b": BranchAndBound
    # }
    #
    # RelaxedSolver = relaxed_solvers[CM().get('architecture.relaxed')]
    # Discretiser = discretisers[CM().get('architecture.discretiser')]
    #
    # dataloader = DynamicDataloader(batch_size=CM().get('training.batch_size'), shuffle=False)
    # dataloader.load_data(CM().get('dataset_files'))
    #
    # relaxed_solver = RelaxedSolver(dataloader)
    # if isinstance(relaxed_solver,BaseTrainableRelaxedSolver):
    #     relaxed_solver.train()
    # prediction_engine = Discretiser(relaxed_solver)
    #
    # if CM().get('training.evaluate'):
    #     for batch_size in [1, 10, 20, 50, 100]:
    #         start_time = time.time()
    #         evaluate(prediction_engine, batch_size)
    #         end_time = time.time()
    #         print(f"Batch size: {batch_size}, time: {end_time - start_time}")

    embedding_manager = EmbeddingManager()
    # embedding_manager.train()
    # embedding_manager.save_embeddings()
    embedding_manager.load_embeddings()
    # print(embedding_manager)

    embedding = torch.tensor([31.49592, 30.524298])
    print(embedding_manager.map_to_material(embedding))
    # print("finished")

    notify()

