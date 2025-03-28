import os
import time
import torch
import wandb
from data.values.Material import Material
from forward.forward_tmm import coating_to_reflective_props
from prediction.BaseTrainableModel import BaseTrainableModel
from prediction.GradientModel import GradientModel
from prediction.CNN import CNN
from prediction.MLP import MLP
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

def notify():
    os.system("echo -ne '\007'")

if __name__ == "__main__":

    models = {
        "gradient": GradientModel,
        "mlp": MLP,
        "cnn": CNN
    }

    if CM().get('wandb.log'):
        print("initialising")
        wandb.init(
            project=CM().get('wandb.project'),
            config=CM().get('wandb.config')
        )

    Model = models[CM().get('architecture')]

    dataloader = DynamicDataloader(batch_size = CM().get('training.batch_size'), shuffle = False)
    dataloader.load_data(CM().get('dataset_files'))

    model = Model(dataloader)
    model.load("out/models/model_guided_switch_1000_3_guided_mlp_single.pt")

    if isinstance(model, BaseTrainableModel):
        model.train()

    if CM().get('training.evaluate'):
        evaluate_model(model)
        test_model(model)


    notify()

