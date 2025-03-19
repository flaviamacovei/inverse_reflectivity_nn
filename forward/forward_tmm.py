import torch
import sys
sys.path.append(sys.path[0] + '/..')
from data.values.ReflectivePropsValue import ReflectivePropsValue
from data.values.Coating import Coating
from utils.ConfigManager import ConfigManager as CM
from tmm_clean.tmm_core import compute_multilayer_optics

def coating_to_reflective_props(coating: Coating):
    thicknesses = coating.get_thicknesses()
    thicknesses = torch.cat([thicknesses, torch.ones((thicknesses.shape[0], 1), device = CM().get('device'))], dim = 1)
    refractive_indices = coating.get_refractive_indices()
    refractive_indices = torch.cat([refractive_indices, torch.ones((refractive_indices.shape[0], 1, CM().get('wavelengths').shape[0]), device = CM().get('device'))], dim = 1)
    
    reflective_props_tensor = compute_multilayer_optics(CM().get('polarisation'), refractive_indices, thicknesses, CM().get('theta'), CM().get('wavelengths'), device = CM().get('device'))['R']

    steps = reflective_props_tensor.shape[:1] + reflective_props_tensor.shape[2:]
    reflective_props_tensor = reflective_props_tensor.reshape(steps)

    result = ReflectivePropsValue(reflective_props_tensor)

    return result