import torch
from torch.utils.data import TensorDataset
from CompletePropsGenerator import CompletePropsGenerator
from MaskedPropsGenerator import MaskedPropsGenerator
from ExplicitPropsGenerator import ExplicitPropsGenerator
import sys
sys.path.append(sys.path[0] + '/..')
from utils.ConfigManager import ConfigManager as CM

def save_tensors_free(generated):
    feature_tensors = []
    for reflective_props in generated:
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
        values = torch.cat((coating.get_thicknesses(), coating.get_refractive_indices()), dim=0).squeeze()
        label_tensor = torch.zeros((CM().get('layers.max') * 2), device=CM().get('device')).float()
        label_tensor.put_(torch.tensor(range(values.shape[0]), device=CM().get('device')), values)
        feature_tensors.append(feature_tensor)
        label_tensors.append(label_tensor)
    feature_tensors = torch.stack(feature_tensors)
    label_tensors = torch.stack(label_tensors)
    return TensorDataset(feature_tensors, label_tensors)

def generate_dataset(generators, save_functions):
    save_function = save_functions[CM().get('data_generation.guidance')]

    num_points = CM().get('data_generation.dataset_size')
    dataset_filename = f"../datasets/{CM().get('data_generation.guidance')}_{CM().get('data_generation.density')}_{num_points}.pt"
    print(f"Generating dataset with {num_points} points")

    Generator = generators[CM().get('data_generation.density')]
    dataset_generator = Generator(num_points)

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
    generate_dataset(generators, save_functions)

