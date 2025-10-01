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
    ownpath = os.path.realpath(__file__)

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
        "air_pad": CM().get('air_pad'),
        "stratified_sampling": CM().get('stratified_sampling'),
        "tolerance": CM().get('tolerance'),
        "density": density,
        "num_points": num_points
    }
    own_path = os.path.realpath(__file__)
    metadata_path = os.path.join(os.path.dirname(os.path.dirname(own_path)), "data/datasets/metadata.yaml")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            content = yaml.safe_load(f)
            # search for properties dictionary match in metadata
            for dataset in content["datasets"]:
                if dataset["properties"] == props_dict:
                    return os.path.join(os.path.dirname(os.path.dirname(ownpath)), dataset["title"])
    return None

def get_props_dict(attributes: dict = None):
    """
    read properties from config and augment with attributes if provided
    """
    # properties read from config
    props_dict = {
        "architecture": CM().get('architecture'),
        "model_details": CM().get(CM().get('architecture')),
        "num_layers": CM().get('num_layers'),
        "min_wl": CM().get('wavelengths')[0].item(),
        "max_wl": CM().get('wavelengths')[-1].item(),
        "wl_step": len(CM().get('wavelengths')),
        "polarisation": CM().get('polarisation'),
        "materials_hash": EM().hash_materials(),
        "num_materials": len(CM().get('materials.thin_films')),
        "theta": CM().get('theta').item(),
        "air_pad": CM().get('air_pad'),
        "stratified_sampling": CM().get('stratified_sampling'),
        "tolerance": CM().get('tolerance'),
        "num_points": CM().get('training.dataset_size'),
        "epochs": CM().get('training.num_epochs'),
        "curriculum": CM().get('training.curriculum')
    }
    # architecture is special: it also is the reference to the model_details
    if "architecture" in attributes.keys():
        props_dict['model_details'] = CM().get(attributes['architecture'])
    if attributes is not None:
        # augment with attributes
        for key, value in attributes.items():
            if key in props_dict.keys() and isinstance(value, dict) and isinstance(props_dict[key], dict):
                # key existing in both dictionaries and values are dictionaries
                # merge dictionaries where value from attributes takes precedence
                props_dict[key] = {**props_dict[key], **value}
            else:
                # key not existing in props_dict / value not of type dictionary
                #               add              /          overwrite
                props_dict[key] = value
    return props_dict

def get_saved_model_path(attributes = None):
    props_dict = get_props_dict(attributes)
    props_dict = round_curriculum(props_dict)
    ownpath = os.path.realpath(__file__)
    models_data_file = os.path.join(os.path.dirname(os.path.dirname(ownpath)), "out/models/models_metadata.yaml")

    if os.path.exists(models_data_file):
        with open(models_data_file, "r") as f:
            content = yaml.safe_load(f)
            # search for properties dictionary match in metadata
            for model in content["models"]:
                model_props = model['properties']
                if 'curriculum' in model_props.keys():
                    model_props = round_curriculum(model_props)
                if model_props == props_dict:
                    return os.path.join(os.path.dirname(os.path.dirname(ownpath)), model["title"])
    return None

def round_curriculum(props_dict: dict):
    props_dict['curriculum'] = [{'density': item['density'], 'percent': round(item['percent'], 4)} for item in props_dict['curriculum']]
    return props_dict

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
    # TODO: can I replace this line with a reference in yaml that is preserved during load and dump?
    config_dict['num_layers'] = configs[config_id]['num_layers']
    config_dict['materials']['substrate'] = configs[config_id]['substrate']
    config_dict['materials']['air'] = configs[config_id]['air']
    config_dict['materials']['thin_films'] = configs[config_id]['materials']
    with open('config.yaml', 'w') as f:
        yaml.dump(config_dict, f, sort_keys = False)