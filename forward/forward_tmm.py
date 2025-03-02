import sys
sys.path.append(sys.path[0] + '/..')
from data.values.ReflectivePropsValue import ReflectivePropsValue
from data.values.Coating import Coating
from utils.ConfigManager import ConfigManager as CM
from tmm_clean.tmm_core import compute_multilayer_optics

def coating_to_reflective_props(coating: Coating):
    thicknesses = coating.get_thicknesses()
    refractive_indices = coating.get_refractive_indices()

    reflective_props_tensor = compute_multilayer_optics('s', refractive_indices, thicknesses, CM().get('theta'), CM().get('wavelengths'), device = CM().get('device'))['R']

    steps = reflective_props_tensor.shape[:1] + reflective_props_tensor.shape[2:]
    reflective_props_tensor = reflective_props_tensor.reshape(steps)

    result = ReflectivePropsValue(reflective_props_tensor)

    return result
