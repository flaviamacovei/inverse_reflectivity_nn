import torch
import sys
sys.path.append(sys.path[0] + '/..')
from data.values.ReflectivityValue import ReflectivityValue
from data.values.Coating import Coating
from utils.ConfigManager import ConfigManager as CM
from tmm_clean.tmm_core import compute_multilayer_optics

def coating_to_reflectivity(coating: Coating):
    """
    Run forward tmm function on coating object to obtain reflectivity value object.

    The tmm function computes the reflectivity and transmission given
        - a polarisation.
        - a tensor of refractive indices. Shape: (|coating|, |wavelengths|)
        - a tensor of thicknesses. Shape: (|coating|)
        - an angle of incidence.
        - a linspace representing the wavelengths of interest.

    Args:
        coating: Coating object.

    Returns:
        Reflectivity value object resulting from tmm.
    """
    thicknesses = coating.get_thicknesses()
    refractive_indices = coating.get_refractive_indices()

    # obtain reflectivity from tmm function
    reflectivity = compute_multilayer_optics(CM().get('polarisation'), refractive_indices, thicknesses, CM().get('theta'), CM().get('wavelengths'), device = CM().get('device'))

    steps = reflectivity.shape[:1] + reflectivity.shape[2:]
    reflectivity = reflectivity.reshape(steps)
    result = ReflectivityValue(reflectivity)
    return result