import torch
import sys
sys.path.append(sys.path[0] + '/..')
from data.values.ReflectivePropsPattern import ReflectivePropsPattern
from config import wavelengths, tolerance, device

class FileInput():
    def __init__(self):
        self.wavelenghts = None
        self.values = None

    def read_from_csv(self, file: str):
        assert file is not None
        assert file.endswith(".csv")

        with open(file, encoding = "utf-8-sig") as f:
            lines = f.readlines()

        self.wavelenghts = [float(line.split(",")[0]) for line in lines]
        self.values = [float(line.split(",")[1]) / 100 for line in lines]
        print(self.values)

    def to_reflective_props_pattern(self):
        lower_bound = torch.tensor(self.values, device = device)
        upper_bound = lower_bound.clone()

        lower_bound = torch.clamp(lower_bound - tolerance / 2, 0, 1)
        upper_bound = torch.clamp(upper_bound + tolerance / 2, 0, 1)

        return ReflectivePropsPattern(lower_bound, upper_bound)