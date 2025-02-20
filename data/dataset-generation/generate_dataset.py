import torch
from torch.utils.data import TensorDataset
from CompletePropsGenerator import CompletePropsGenerator
import sys

def generate_dataset(num_points):
    print(f"Generating dataset with {num_points} points")
    train_set_generator = CompletePropsGenerator(num_points)
    train_tensors = []
    for reflective_props in train_set_generator.generate():
        train_tensor = torch.cat((reflective_props.get_lower_bound(), reflective_props.get_upper_bound()))
        train_tensors.append(train_tensor)

    train_tensors = torch.stack(train_tensors)

    train_set = TensorDataset(train_tensors)
    torch.save(train_set, f'C:/Users/diama/Desktop/uni/winter24/research/inverse_reflectivity_nn/data/datasets/complete_props_{num_points}.pt')

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_dataset.py <num_points>")
    else:
        num_points = int(sys.argv[1])
        generate_dataset(num_points)
        print(f"Dataset saved to datasets/complete_props_{num_points}.pt")

