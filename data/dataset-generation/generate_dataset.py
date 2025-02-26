import torch
from torch.utils.data import TensorDataset
from CompletePropsGenerator import CompletePropsGenerator
import sys
sys.path.append(sys.path[0] + "/..")
from config import device, max_num_layers


def generate_dataset(num_points):
    print(f"Generating dataset with {num_points} points")
    train_set_generator = CompletePropsGenerator(num_points)
    feature_tensors = []
    label_tensors = []
    for (reflective_props, coating) in train_set_generator.generate():
        feature_tensor = torch.cat((reflective_props.get_lower_bound(), reflective_props.get_upper_bound()), dim = 1).squeeze()
        values = torch.cat((coating.get_thicknesses(), coating.get_refractive_indices()), dim = 0).squeeze()
        label_tensor = torch.zeros((max_num_layers * 2), device = device).float()
        label_tensor.put_(torch.tensor(range(values.shape[0]), device = device), values)
        feature_tensors.append(feature_tensor)
        label_tensors.append(label_tensor)

    feature_tensors = torch.stack(feature_tensors)
    label_tensors = torch.stack(label_tensors)

    train_set = TensorDataset(feature_tensors, label_tensors)
    torch.save(train_set, f'../datasets/complete_with_labels_{num_points}.pt')

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_dataset.py <num_points>")
    else:
        num_points = int(sys.argv[1])
        generate_dataset(num_points)
        print(f"Dataset saved to datasets/complete_props_{num_points}.pt")

