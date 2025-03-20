import torch
from torch.utils.data import TensorDataset
import sys
sys.path.append(sys.path[0] + '/..')
from utils.ConfigManager import ConfigManager as CM
from data.material_embedding.EmbeddingManager import EmbeddingManager as EM
from data.dataset_generation.CompletePropsGenerator import CompletePropsGenerator
from data.dataset_generation.MaskedPropsGenerator import MaskedPropsGenerator
from data.dataset_generation.ExplicitPropsGenerator import ExplicitPropsGenerator

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
    dataset_filename = f"data/datasets/{CM().get('data_generation.guidance')}_{CM().get('data_generation.density')}_{num_points}_{EM().hash_materials()}.pt"
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

