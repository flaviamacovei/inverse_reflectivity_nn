import torch
from torch.utils.data import TensorDataset
import yaml
import os
import sys
sys.path.append(sys.path[0] + '/..')
from utils.ConfigManager import ConfigManager as CM
from data.material_embedding.EmbeddingManager import EmbeddingManager as EM
from data.dataset_generation.CompletePropsGenerator import CompletePropsGenerator
from data.dataset_generation.MaskedPropsGenerator import MaskedPropsGenerator
from data.dataset_generation.ExplicitPropsGenerator import ExplicitPropsGenerator
from utils.os_utils import short_hash, get_unique_filename

def save_tensors_free(generated):
    feature_tensors = []
    for (reflective_props, _) in generated:
        feature_tensor = torch.cat((reflective_props.get_lower_bound(), reflective_props.get_upper_bound()),
                                   dim=1).squeeze()
        feature_tensors.append(feature_tensor)
    feature_tensors = torch.stack(feature_tensors)
    return TensorDataset(feature_tensors)

def save_tensors_guided(generated):
    feature_tensors = []
    label_tensors = []
    for (reflective_props, coating) in generated:
        feature_tensor = torch.cat((reflective_props.get_lower_bound(), reflective_props.get_upper_bound()),
                                   dim=1).squeeze()
        label_tensor = coating.get_encoding().squeeze()
        feature_tensors.append(feature_tensor)
        label_tensors.append(label_tensor)
    feature_tensors = torch.stack(feature_tensors)
    label_tensors = torch.stack(label_tensors)
    return TensorDataset(feature_tensors, label_tensors)

def generate_dataset(generator, save_function):

    num_points = CM().get('data_generation.dataset_size')
    FILEPATH_METADATA = "data/datasets/metadata.yaml"
    props_dict = {
        "partition": CM().get('data_generation.partition'),
        "num_layers": CM().get('num_layers'),
        "min_wl": CM().get('wavelengths')[0].item(),
        "max_wl": CM().get('wavelengths')[-1].item(),
        "wl_step": len(CM().get('wavelengths')),
        "polarisation": CM().get('polarisation'),
        "materials_hash": EM().hash_materials(),
        "theta": CM().get('theta').item(),
        "tolerance": CM().get('tolerance'),
        "guidance": CM().get('data_generation.guidance'),
        "density": CM().get('data_generation.density'),
        "num_points": num_points
    }
    dataset_hash = short_hash(props_dict)
    dataset_filename = get_unique_filename(f"data/datasets/dataset_{dataset_hash}.pt")
    if not os.path.exists(FILEPATH_METADATA):
        with open(FILEPATH_METADATA, "w") as f:
            f.write("datasets:\n")
    with open(FILEPATH_METADATA, "r+") as f:
        content = yaml.safe_load(f)
        content["datasets"].append({**{"title": dataset_filename}, **{"properties": props_dict}})
        f.seek(0)
        yaml.dump(content, f, sort_keys = False, default_flow_style=False, indent = 2)
    print(f"Generating dataset with {num_points} points")

    dataset_generator = generator(num_points)

    generated_data = dataset_generator.generate()
    dataset = save_function(generated_data)
    torch.save(dataset, dataset_filename)
    print(f"Dataset saved to {dataset_filename}")

if __name__ == "__main__":
    generators = {
        "complete": CompletePropsGenerator,
        "masked": MaskedPropsGenerator,
        "explicit": ExplicitPropsGenerator
    }
    save_functions = {
        "free": save_tensors_free,
        "guided": save_tensors_guided
    }
    generator = generators[CM().get('data_generation.density')]
    save_function = save_functions[CM().get('data_generation.guidance')]
    generate_dataset(generator, save_function)

