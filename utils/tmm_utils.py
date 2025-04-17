import torch
import sys
sys.path.append(sys.path[0] + '/..')
from utils.ConfigManager import ConfigManager as CM

def get_wavelength_index(wavelength):
    """Find index in wavelengths at which value is closest to specified wavelength."""
    FLOAT_TOLERANCE = 1e-4
    # find index with closest value
    closest_index = torch.argmin(torch.abs(CM().get('wavelengths') - wavelength))
    # check if value is within tolerance
    if abs(CM().get('wavelengths')[closest_index] - wavelength) < FLOAT_TOLERANCE:
        return closest_index
    # if no close match is found, return None
    return None