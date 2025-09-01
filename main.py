import os
import pandas as pd
import time
import torch
from torch.utils.data import DataLoader
import wandb
import yaml
from itertools import product
import time
from data.values.BaseMaterial import BaseMaterial
from forward.forward_tmm import coating_to_reflectivity
from prediction.BaseTrainableModel import BaseTrainableModel
from prediction.CNN import CNN
from prediction.GradientModel import GradientModel
from prediction.MLP import MLP
from prediction.RandomModel import RandomModel
from prediction.Transformer import Transformer
from prediction.Hybrid import Hybrid
from data.dataloaders.DynamicDataloader import DynamicDataloader
from data.values.Coating import Coating
from ui.visualise import visualise
from ui.FileInput import FileInput
from generate_dataset import generate_dataset
from data.values.ConstantRIMaterial import ConstantRIMaterial
from data.values.ReflectivityPattern import ReflectivityPattern
from utils.ConfigManager import ConfigManager as CM
from evaluation.model_eval import evaluate_model, test_model
from data.material_embedding.EmbeddingManager import EmbeddingManager as EM
from data.dataset_generation.CompletePatternGenerator import CompletePatternGenerator
from data.dataset_generation.MaskedPatternGenerator import MaskedPatternGenerator
from data.dataset_generation.ExplicitPatternGenerator import ExplicitPatternGenerator
from tmm_clean.tmm_core import compute_multilayer_optics
from utils.data_utils import get_saved_model_path

def notify():
    os.system("echo -ne '\007'")

def main():
    dataloader = DynamicDataloader(batch_size = 1, shuffle = False)
    dataloader.load_leg(0)
    point = dataloader[0]
    reflectivity = point[0].to(CM().get('device'))
    lower_bound, upper_bound = reflectivity.chunk(2, dim = -1)
    # encodings = point[1].to(CM().get('device'))
    # print(encodings)
    # coating = Coating(encodings[None])

    # fi = FileInput()
    # fi.read_from_csv("data/data_files/Neuschwanstein_target.csv")
    # target = fi.to_reflectivity_pattern()


    target = ReflectivityPattern(lower_bound[None], upper_bound[None])
    model = MLP()
    model.load_or_train()
    coating = model.predict(target)
    print(coating)
    preds = coating_to_reflectivity(coating)
    visualise(preds = preds, refs = target, filename = "pca")
    # print(f"coating: {coating}")
    # print(f"refractive indices: {coating.get_refractive_indices()}")
    # lower_bound, upper_bound = point[0][None].chunk(2, dim = 1)
    # visualise(refs = ReflectivityPattern(lower_bound, upper_bound), filename = "test")
    notify()

def visualise_test_data(architecture: str = None):
    if architecture is not None:
        model_classes = {
            'random': {'class': RandomModel},
            'gradient': {'class': GradientModel},
            'mlp': {'class': MLP},
            'mlp+gradient': {'class': lambda: Hybrid('mlp')},
            'cnn': {'class': CNN},
            'cnn+gradient': {'class': lambda: Hybrid('cnn')},
            'transformer': {'class': Transformer},
            'transformer+gradient': {'class': lambda: Hybrid('transformer')},
            'transformer_struct_caus_mask+gradient': {'class': lambda: Hybrid('transformer'),
                                                      'attrs': {'src_mask': False, 'tgt_struct_mask': True,
                                                                'tgt_caus_mask': True}},
        }
        ModelClass = model_classes[architecture]['class']
        model = ModelClass()
        if isinstance(model, BaseTrainableModel):
            attrs = {'model_details': model_classes[architecture]['attrs']} if 'attrs' in model_classes[architecture].keys() else None
            model.load_or_train(attrs)
    dataset = torch.load(f"data/datasets/test_data/test_data.pt", weights_only=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    for i, batch in enumerate(dataloader):
        reflectivity = batch[0]
        reflectivity = reflectivity.to(CM().get('device'))
        lower_bound, upper_bound = torch.chunk(reflectivity, 2, -1)
        target = ReflectivityPattern(lower_bound, upper_bound)
        if architecture is not None:
            predicted_coating = model.predict(target)
            preds = coating_to_reflectivity(predicted_coating)
        else:
            preds = None
        visualise(refs=target, preds=preds, filename=f"test_data_{architecture}_2_{i}")

if __name__ == "__main__":
    # overview()
    # dataloader = DynamicDataloader(CM().get('training.dataset_size'), False)
    # # complete
    # dataloader.load_leg(0)
    # reflectivity, coating = dataloader[1]
    # lower_bound, upper_bound = torch.chunk(reflectivity[None], 2, 1)
    # pattern = ReflectivityPattern(lower_bound, upper_bound)
    # label = Coating(coating[None])
    #
    # visualise(refs = pattern, filename = "att_test")
    #
    # model = Transformer()
    # loaded_model_path = get_loaded_model_name("transformer")
    # trainable_model = torch.load(loaded_model_path, weights_only = False)
    # trainable_model = trainable_model.to(CM().get('device'))
    # model.model = trainable_model
    # model.visualise_attention(pattern, label)

    visualise_test_data('transformer_struct_caus_mask+gradient')
    print("<3")