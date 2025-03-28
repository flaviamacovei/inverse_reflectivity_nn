import torch
import sys
sys.path.append(sys.path[0] + '/..')
from data.values.ReflectivePropsPattern import ReflectivePropsPattern
from utils.ConfigManager import ConfigManager as CM
from utils.tmm_utils import get_wavelength_index

class FileInput():
    def __init__(self):
        self.wavelengths = None
        self.values = None

    def read_from_csv(self, file: str):
        assert file is not None
        assert file.endswith(".csv")

        with open(file, encoding = "utf-8-sig") as f:
            lines = f.readlines()

        self.wavelengths = [float(line.split(",")[0]) for line in lines]
        # print(self.wavelengths)
        self.values = [float(line.split(",")[1]) / 100 for line in lines]


    def to_reflective_props_pattern(self):
        # FIXME: use torch
        lower_bound = torch.zeros((1, CM().get('wavelengths').shape[0]), device=CM().get('device'))
        upper_bound = torch.ones((1, CM().get('wavelengths').shape[0]), device=CM().get('device'))

        for i in range(len(self.wavelengths)):
            wl = self.wavelengths[i] / 1e3
            index = get_wavelength_index(wl)
            if index:
                lower_bound[0][index] = self.values[i]
                upper_bound[0][index] = self.values[i]

        lower_bound = torch.clamp(lower_bound - CM().get('tolerance') / 2, 0, 1)
        upper_bound = torch.clamp(upper_bound + CM().get('tolerance') / 2, 0, 1)

        print(f"File converted successfully.")

        return ReflectivePropsPattern(lower_bound, upper_bound)