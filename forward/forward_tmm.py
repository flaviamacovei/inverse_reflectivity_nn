import torch
import sys
sys.path.append(sys.path[0] + '/..')
from data.values.ReflectivePropsValue import ReflectivePropsValue
from data.values.Coating import Coating
from utils.ConfigManager import ConfigManager as CM
from tmm_clean.tmm_core import compute_multilayer_optics

def coating_to_reflective_props(coating: Coating):
    """
    Run forward tmm function on coating object to obtain reflective properties value object.

    The tmm function computes the reflectivity and transmission given
        - a polarisation.
        - a tensor of refractive indices. Shape: (|coating|, |wavelengths|)
        - a tensor of thicknesses. Shape: (|coating|)
        - an angle of incidence.
        - a linspace representing the wavelengths of interest.

    Args:
        coating: Coating object.

    Returns:
        Reflective properties value object resulting from tmm.
    """
    thicknesses = coating.get_thicknesses()
    refractive_indices = coating.get_refractive_indices()

    # obtain reflectivity from tmm function
    reflective_props_tensor = compute_multilayer_optics(CM().get('polarisation'), refractive_indices, thicknesses, CM().get('theta'), CM().get('wavelengths'), device = CM().get('device'))['R']

    steps = reflective_props_tensor.shape[:1] + reflective_props_tensor.shape[2:]
    reflective_props_tensor = reflective_props_tensor.reshape(steps)
    result = ReflectivePropsValue(reflective_props_tensor)
    return result