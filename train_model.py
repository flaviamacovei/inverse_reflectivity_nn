import os
import time
import torch
from itertools import product
import yaml
import wandb
from torch.utils.data import DataLoader
from generate_dataset import generate_dataset
from data.values.BaseMaterial import BaseMaterial
from forward.forward_tmm import coating_to_reflectivity
from prediction.BaseTrainableModel import BaseTrainableModel
from prediction.GradientModel import GradientModel
from prediction.MLP import MLP
from prediction.CNN import CNN
from prediction.RNN import RNN
from prediction.Transformer import Transformer
from data.dataloaders.DynamicDataloader import DynamicDataloader
from data.values.Coating import Coating
from ui.visualise import visualise
from ui.FileInput import FileInput
from data.values.ReflectivityPattern import ReflectivityPattern
from utils.ConfigManager import ConfigManager as CM
from evaluation.model_eval import evaluate_model, test_model
from data.material_embedding.EmbeddingManager import EmbeddingManager as EM
from data.dataset_generation.CompletePatternGenerator import CompletePatternGenerator
from data.dataset_generation.MaskedPatternGenerator import MaskedPatternGenerator
from data.dataset_generation.ExplicitPatternGenerator import ExplicitPatternGenerator
from tmm_clean.tmm_core import compute_multilayer_optics
from score_model import score_model
from structure.StructureAutoEncoder import StructureAutoEncoder, TrainableAutoEncoder

def train_model():
    """Instantiate and train model."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # map architecture specified in config to model classes
    models = {
        "gradient": GradientModel,
        "mlp": MLP,
        "cnn": CNN,
        "rnn": RNN,
        "transformer": Transformer,
    }

    if CM().get('wandb.log') and not CM().get('wandb.sweep'):
        print("initialising weights and biases")
        wandb.init(
            project=CM().get('wandb.project'),
            config=CM().get('wandb.config')
        )

    # select architecture
    Model = models[CM().get('architecture')]
    model = Model()

    print(f"num params: {model.get_num_params()}")

    if isinstance(model, BaseTrainableModel):
        model.train()

    if CM().get('training.evaluate'):
        score_model(model)

if __name__ == "__main__":
    train_model()