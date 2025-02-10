import numpy as np
import torch
from tmm_fast import coh_tmm as tmm
import os
import time
import sys
sys.path.append(sys.path[0] + '/..')
from data.values.ReflectivePropsValue import ReflectivePropsValue
from data.values.Coating import Coating
from mirror_transformer.direct_opt import SM, Material, Stack
from config import device, num_stacks, materials_file

def make_stack():
    sm = SM()
    materials_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "mirror_transformer")
    with open(materials_dir + "/" + materials_file, "r", encoding = "ISO-8859-1") as f:
        for line in f:
            sm.run(line)
    materials = [Material(m) for m in sm.materials]
    st = Stack(sm.front, materials)
    return st

def create_args(start_wl: int, end_wl: int, steps: int):
    # prep
    st = make_stack()

    # make arguments
    wavelenghts = np.linspace(start_wl, end_wl, steps) * (10 ** (-9))
    M = torch.tensor(st.make_M(wavelenghts, num_stacks), dtype = torch.float32).to(device)
    theta = torch.tensor(np.linspace(0, 0, 1) * (np.pi / 180), dtype = torch.float32).to(device)
    wavelengths = torch.tensor(wavelenghts, dtype=torch.float32).to(device) * 1000000

    return M, theta, wavelengths

def coating_to_reflective_props(coating: Coating, start_wl: int, end_wl: int, steps: int):
    thicknesses = coating.get_thicknesses()

    M, theta, wavelengths = create_args(start_wl, end_wl, steps)
    reflective_props_tensor = tmm('p', M, (thicknesses), theta, wavelengths, device=device)['R']

    steps = reflective_props_tensor.shape[2]
    reflective_props_tensor = reflective_props_tensor.reshape(steps)

    result = ReflectivePropsValue(start_wl, end_wl, reflective_props_tensor)

    return result
