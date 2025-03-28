import torch
import sys
sys.path.append(sys.path[0] + '/..')
from utils.ConfigManager import ConfigManager as CM

def get_wavelength_index(wavelength):
    FLOAT_TOLERANCE = 1e-4
    closest_index = torch.argmin(torch.abs(CM().get('wavelengths') - wavelength))
    if abs(CM().get('wavelengths')[closest_index] - wavelength) < FLOAT_TOLERANCE:
        return closest_index
    return None