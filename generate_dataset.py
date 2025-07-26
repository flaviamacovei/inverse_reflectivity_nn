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
from utils.data_utils import get_dataset_name


def save_tensors(generated):
    """
    Convert a list of tensors to a TensorDataset.

    Args:
        generated: A list of tuples of the form (reflective properties pattern, coating).
    """
    feature_tensors = []
    label_tensors = []
    for (reflective_props, coating) in generated:
        # convert reflective properties to concatenated tensor
        feature_tensor = torch.cat((reflective_props.get_lower_bound(), reflective_props.get_upper_bound()),
                                   dim=1).squeeze()
        if coating is None:
            # for explicit dataset, no coating is provided so use dummy data
            label_tensor = torch.zeros(feature_tensor.shape[0])
        else:
            # for complete and masked datasets, coating is provided
            label_tensor = coating.get_encoding().squeeze()
        feature_tensors.append(feature_tensor)
        label_tensors.append(label_tensor)
    feature_tensors = torch.cat(feature_tensors, dim = 0)
    label_tensors = torch.cat(label_tensors, dim = 0)
    return TensorDataset(feature_tensors, label_tensors)


def write_to_metadata(dataset_filename, props_dict):
    """Append dataset properties to metadata file."""
    FILEPATH_METADATA = "data/datasets/metadata.yaml"
    if not os.path.exists(FILEPATH_METADATA):
        # create file if it does not exist
        with open(FILEPATH_METADATA, "w") as f:
            yaml.dump({"datasets": [{**{'title': dataset_filename}, **{'properties': props_dict}}]}, f, sort_keys=False,
                      default_flow_style=False, indent=2)
    else:
        # load, append, and write to file
        with open(FILEPATH_METADATA, "r+") as f:
            content = yaml.safe_load(f)
            content["datasets"].append({**{"title": dataset_filename}, **{"properties": props_dict}})
            f.seek(0)
            yaml.dump(content, f, sort_keys=False, default_flow_style=False, indent=2)

def dataset_exists(split: str):
    for density in ['complete', 'masked', 'explicit']:
        dataset_name = get_dataset_name(split, density)
        if not dataset_name:
            return False
    return True


def generate_dataset(generators, split):
    """
    Generate data and save to file.

    Args:
        generators: A dictionary mapping density to the generator class.
        split: "training" or "validation".
    """
    if split == "training":
        # number of points in training dataset specified in config file
        num_points = CM().get('training.dataset_size')
    else:
        # all validation datasets have 100 points
        num_points = 100
    if dataset_exists(split):
        print("No generation necessary. Dataset already exists.")
        return
    # maximum total size of a dataset that cuda memory can handle
    MAX_SIZE = 2_400_000
    # maximum number of points that cuda memory can handle
    MAX_POINTS_PER_SEGMENT = MAX_SIZE // CM().get('wavelengths').shape[0]
    densities = list(generators.keys())
    for density in densities:
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
        dataset_hash = short_hash(props_dict)
        dataset_filename = get_unique_filename(f"data/datasets/dataset_{dataset_hash}.pt")
        print(f"Generating {density} dataset with {num_points} points")
        Generator = generators[density]

        if num_points <= MAX_POINTS_PER_SEGMENT:
            # dataset small enough that no segmentation is needed
            dataset_generator = Generator(num_points)

            generated_data = dataset_generator.generate()
            dataset = save_tensors(generated_data)
            torch.save(dataset, dataset_filename)
        else:
            # segmentation needed
            for i in range(num_points // MAX_POINTS_PER_SEGMENT + 1):
                num_points_segment = min(num_points - i * MAX_POINTS_PER_SEGMENT, MAX_POINTS_PER_SEGMENT)
                dataset_generator = Generator(num_points_segment)
                generated_data = dataset_generator.generate()
                dataset = save_tensors(generated_data)
                torch.save(dataset, dataset_filename[:-3] + f"_seg_{i + 1}.pt")
                # clear memory
                del dataset
                torch.cuda.empty_cache()
        # append dataset properties to metadata after successful generation
        write_to_metadata(dataset_filename, props_dict)
        print(f"Dataset saved to {dataset_filename}")

if __name__ == "__main__":
    # default split (no argument specified) is "training"
    split = "training" if len(sys.argv) == 1 else sys.argv[1]
    generators = {
        "complete": CompletePropsGenerator,
        "masked": MaskedPropsGenerator,
        "explicit": ExplicitPropsGenerator
    }
    generate_dataset(generators, split)

