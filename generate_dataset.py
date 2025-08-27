import torch
from torch.utils.data import TensorDataset, WeightedRandomSampler, DataLoader
import yaml
import os
import sys
sys.path.append(sys.path[0] + '/..')
from utils.ConfigManager import ConfigManager as CM
from data.material_embedding.EmbeddingManager import EmbeddingManager as EM
from data.dataset_generation.CompletePatternGenerator import CompletePatternGenerator
from data.dataset_generation.MaskedPatternGenerator import MaskedPatternGenerator
from data.dataset_generation.ExplicitPatternGenerator import ExplicitPatternGenerator
from utils.os_utils import short_hash, get_unique_filename
from utils.data_utils import get_dataset_name


def convert_to_dataset(generated):
    """
    Convert a list of tensors to a TensorDataset.

    Args:
        generated: A list of tuples of the form (ReflectivityPattern, Coating).
    """
    feature_tensors = []
    label_tensors = []
    for (reflectivity, coating) in generated:
        # convert reflectivity to concatenated tensor
        feature_tensor = torch.cat((reflectivity.get_lower_bound(), reflectivity.get_upper_bound()),
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
    print(f"from convert: {feature_tensors.device}, {label_tensors.device}")
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

def sample(data: TensorDataset, num_points: int):
    """
    If stratified_sampling is true, this function will be called.
    Generated data is OVERSAMPLING_FACTOR times larger than needed.
    Divide into bins and sample evenly many points per bin.
    """
    BINS = 10
    reflectivity, coating = data.tensors
    averaged = torch.sum(reflectivity, dim = -1) / reflectivity.shape[-1]
    hist = torch.histc(averaged, bins = BINS, min = 0, max = 1)

    # weights are inversely proportional to frequency in data
    # small epsilon to avoid division by 0
    weights = 1 / (hist + 0.001)
    weights[torch.eq(hist, 0)] = 0
    # normalise
    weights = weights / torch.sum(weights)

    # assign weight to point: if it is inside bracket (greater equal to lower bound, less than upper bound)
    step_size = 1 / BINS
    lower_bound = torch.linspace(0, 1 - step_size, BINS, device = CM().get('device'))
    upper_bound = torch.linspace(step_size, 1, BINS, device = CM().get('device'))
    print(f"averaged device: {averaged.device}")
    print(f"bound device. {lower_bound.device}, {upper_bound.device}")
    greater_equal = torch.ge(averaged[:, None].repeat(1, BINS), lower_bound)
    less_than = torch.lt(averaged[:, None].repeat(1, BINS), upper_bound)
    mask = torch.logical_and(greater_equal, less_than)
    masked_weights = weights[None].repeat(reflectivity.shape[0], 1) * mask
    point_weights = masked_weights.sum(dim=1)
    # possibly num_samples to data.shape[0]
    sampler = WeightedRandomSampler(weights = point_weights, num_samples = num_points, replacement = False)
    dataloader = DataLoader(data, sampler = sampler, batch_size = num_points)
    return TensorDataset(next(iter(dataloader))[0], next(iter(dataloader))[1])

def generate_dataset(split):
    """
    Generate data and save to file.

    Args:
        generators: A dictionary mapping density to the generator class.
        split: "training" or "validation".
    """
    generators = {
        "complete": CompletePatternGenerator,
        "masked": MaskedPatternGenerator,
        "explicit": ExplicitPatternGenerator
    }
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
    # scale factor needed for stratified sampling
    OVERSAMPLING_FACTOR = 100
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
            "air_pad": CM().get('air_pad'),
            "stratified_sampling": CM().get('stratified_sampling'),
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
            generation_points = num_points * OVERSAMPLING_FACTOR if CM().get('stratified_sampling') else num_points
            dataset_generator = Generator(generation_points)

            generated_data = dataset_generator.generate()
            dataset = convert_to_dataset(generated_data)
            if CM().get('stratified_sampling'):
                dataset = sample(dataset, num_points = num_points)
            torch.save(dataset, dataset_filename)
        else:
            # segmentation needed
            for i in range(num_points // MAX_POINTS_PER_SEGMENT + 1):
                num_points_segment = min(num_points - i * MAX_POINTS_PER_SEGMENT, MAX_POINTS_PER_SEGMENT)
                generation_points = num_points_segment * OVERSAMPLING_FACTOR if CM().get('stratified_sampling') else num_points_segment
                dataset_generator = Generator(generation_points)
                generated_data = dataset_generator.generate()
                dataset = convert_to_dataset(generated_data)
                if CM().get('stratified_sampling'):
                    dataset = sample(dataset, num_points = num_points_segment)
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
    generate_dataset(split)

