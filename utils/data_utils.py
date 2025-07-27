import yaml
import sys
import os
sys.path.append(sys.path[0] + '/..')
from utils.ConfigManager import ConfigManager as CM
from data.material_embedding.EmbeddingManager import EmbeddingManager as EM

def get_dataset_name(split: str, density: str):
    """
    Return filename for dataset with specified split and density and settings matching config file, if it exists.

    Args:
        split: "training" or "validation".
        density: "complete", "masked", or "explicit".
    """
    if split == "validation":
        # all validation datasets have 100 points
        num_points = 100
    else:
        # number of points in training dataset specified in config file
        num_points = CM().get('training.dataset_size')
    # create properties dictionary by which to identify dataset
    props_dict = {
        "split": split,
        "num_layers": CM().get('num_layers'),
        "min_wl": CM().get('wavelengths')[0].item(),
        "max_wl": CM().get('wavelengths')[-1].item(),
        "wl_step": len(CM().get('wavelengths')),
        "polarisation": CM().get('polarisation'),
        "materials_hash": EM().hash_materials(),
        "theta": CM().get('theta').item(),
        "tolerance": CM().get('tolerance'),
        "density": density,
        "num_points": num_points
    }
    if os.path.exists("data/datasets/metadata.yaml"):
        with open("data/datasets/metadata.yaml", "r") as f:
            content = yaml.safe_load(f)
            # search for properties dictionary match in metadata
            for dataset in content["datasets"]:
                if dataset["properties"] == props_dict:
                    return dataset["title"]
    return None


def load_config(config_id: int):
    configs = {
        1: {
            "num_layers": 5,
            "substrate": "Suprasil Heraeus",
            "air": "Air",
            "materials": ["SiO2 IBS", "TiO2 IBS"]
        },
        2: {
            "num_layers": 7,
            "substrate": "DC_Substrate",
            "air": "DC_Air",
            "materials": ["H", "L", "A", "F", "M", "T", "Ag", "Au", "Ni"]
        },
        3: {
            "num_layers": 15,
            "substrate": "DC_Substrate",
            "air": "DC_Air",
            "materials": ["H", "L", "A", "F", "M", "T", "Ag", "Au", "Ni"],
        }
    }

    assert config_id in [1, 2, 3], "Available config ids: 1, 2, 3"
    with open('config.yaml', 'r') as f:
        config_dict = yaml.safe_load(f)
    config_dict['layers']['max'] = configs[config_id]['num_layers']
    config_dict['materials']['substrate'] = configs[config_id]['substrate']
    config_dict['materials']['air'] = configs[config_id]['air']
    config_dict['materials']['thin_films'] = configs[config_id]['materials']
    with open('config.yaml', 'w') as f:
        yaml.dump(config_dict, f, sort_keys = False)