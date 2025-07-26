import os
import pandas as pd
import time
import torch
import wandb
import yaml
from itertools import product
import time
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
from generate_dataset import generate_dataset
from data.values.ConstantRIMaterial import ConstantRIMaterial
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
    # target = fi.to_reflective_props_pattern()


    target = ReflectivePropsPattern(lower_bound[None], upper_bound[None])
    model = MLP()
    model.load("out/models/model_33543.pt")
    coating = model.predict(target)
    print(coating)
    preds = coating_to_reflective_props(coating)
    visualise(preds = preds, refs = target, filename = "pca")
    # print(f"coating: {coating}")
    # print(f"refractive indices: {coating.get_refractive_indices()}")
    # lower_bound, upper_bound = point[0][None].chunk(2, dim = 1)
    # visualise(refs = ReflectivePropsPattern(lower_bound, upper_bound), filename = "test")
    notify()

def overview():
    overview = pd.DataFrame(columns = ["prediction type", "num layers", "num materials", "validation error", "test error", "filename"])
    prediction_types = ["mlp", "cnn"]
    num_layers = range(1, 23)
    num_materials = range(1, 10)
    combinations = list(product(prediction_types, num_layers, num_materials))
    overview["prediction type"] = [combination[0] for combination in combinations]
    overview["num layers"] = [combination[1] for combination in combinations]
    overview["num materials"] = [combination[2] for combination in combinations]

    MODEL_METADATA = "out/models/models_metadata.yaml"
    if not os.path.exists(MODEL_METADATA):
        raise FileNotFoundError(f"No file {MODEL_METADATA}")
    models = {
        "mlp": MLP,
        "cnn": CNN,
        "transformer": Transformer
    }

    for idx, row in overview.iterrows():
        rowdict = dict(row)
        CM().set_layers_to(rowdict['num layers'])
        props_dict = {
            "architecture": rowdict["prediction type"],
            "num_layers": rowdict["num layers"],
            "min_wl": CM().get('wavelengths')[0].item(),
            "max_wl": CM().get('wavelengths')[-1].item(),
            "wl_step": len(CM().get('wavelengths')),
            "polarisation": CM().get('polarisation'),
            "num_materials": rowdict["num materials"],
            "theta": CM().get('theta').item(),
            "tolerance": CM().get('tolerance'),
            "num_points": CM().get('training.dataset_size')
        }
        filename = ''
        with open(MODEL_METADATA, "r") as f:
            content = yaml.safe_load(f)
            for model in content['models']:
                if set(props_dict.items()).issubset(model['properties'].items()):
                    filename = model['title']
                    break
        if filename == '':
            continue

        Model = models[rowdict['prediction type']]
        model = Model()
        model.load(filename, weights_only = False)
        try:
            validation_errors = evaluate_model(model)
        except FileNotFoundError:
            split = "validation"
            generators = {
                "complete": CompletePropsGenerator,
                "masked": MaskedPropsGenerator,
                "explicit": ExplicitPropsGenerator
            }
            generate_dataset(generators, split)
            validation_errors = evaluate_model(model)
        validation_error = sum(validation_errors)
        test_error = test_model(model)
        overview.loc[idx, 'validation error'] = validation_error
        overview.loc[idx, 'test error'] = test_error
        overview.loc[idx, 'filename'] = filename

    overview.dropna(axis = 0, subset = ['validation error', 'test error', 'filename'], inplace = True)

    overview.to_csv("out/overview.csv")


if __name__ == "__main__":
    overview()
