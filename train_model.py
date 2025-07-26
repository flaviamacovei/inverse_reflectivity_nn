import os
import time
import torch
from itertools import product
import yaml
import wandb
from torch.utils.data import DataLoader
from generate_dataset import generate_dataset
from data.values.Material import Material
from forward.forward_tmm import coating_to_reflective_props
from prediction.BaseTrainableModel import BaseTrainableModel
from prediction.CNN import CNN
from prediction.MLP import MLP
from prediction.Transformer import Transformer
from data.dataloaders.DynamicDataloader import DynamicDataloader
from data.values.Coating import Coating
from ui.visualise import visualise
from ui.FileInput import FileInput
from data.values.ReflectivePropsPattern import ReflectivePropsPattern
from utils.ConfigManager import ConfigManager as CM
from evaluation.model_eval import evaluate_model, test_model
from data.material_embedding.EmbeddingManager import EmbeddingManager as EM
from data.dataset_generation.CompletePropsGenerator import CompletePropsGenerator
from data.dataset_generation.MaskedPropsGenerator import MaskedPropsGenerator
from data.dataset_generation.ExplicitPropsGenerator import ExplicitPropsGenerator
from tmm_clean.tmm_core import compute_multilayer_optics
from score_model import score_model

def train_model():
    """Instantiate and train model."""
    # map architecture specified in config to model classes
    models = {
        "mlp": MLP,
        "cnn": CNN,
        "transformer": Transformer
    }

    if CM().get('wandb.log'):
        print("initialising weights and biases")
        wandb.init(
            project=CM().get('wandb.project'),
            config=CM().get('wandb.config')
        )

    # select architecture
    Model = models[CM().get('architecture')]
    model = Model()

    if isinstance(model, BaseTrainableModel):
        model.train()

    if CM().get('training.evaluate'):
        score_model(model)

if __name__ == "__main__":
    num_layers = range(1, 23)
    for num in num_layers:
        CM().set_layers_to(num)
        print(f"\n\n\n{num} layer{'s' if num > 1 else ''}")
        try:
            train_model()
        except FileNotFoundError:
            split = "training"
            generators = {
                "complete": CompletePropsGenerator,
                "masked": MaskedPropsGenerator,
                "explicit": ExplicitPropsGenerator
            }
            generate_dataset(generators, split)
            train_model()
