import numpy as np
import torch
from tmm_fast import coh_tmm as tmm
import sys
sys.path.append(sys.path[0] + '/..')
from data.values.ReflectiveProps import ReflectiveProps
from data.values.Coating import Coating
from mirror_transformer.direct_opt import SM, Material, Stack
from config import device, num_stacks, materials_file

device = 'cpu'

def make_stack():
    sm = SM()
    with open("../mirror_transformer/" + materials_file, "r", encoding = "ISO-8859-1") as f:
        for line in f:
            sm.run(line)
    materials = [Material(m) for m in sm.materials]
    st = Stack(sm.front, materials)
    return st

def create_args(reflective_props: ReflectiveProps):
    # prep
    st = make_stack()

    # make arguments
    wavelenghts = np.linspace(reflective_props.get_start_wl(), reflective_props.get_end_wl(), reflective_props.get_steps()) * (10 ** (-9))
    M = torch.tensor(st.make_M(wavelenghts, num_stacks), dtype = torch.float32).to(device)
    theta = torch.tensor(np.linspace(0, 0, 1) * (np.pi / 180), dtype = torch.float32).to(device)
    wavelengths = torch.tensor(wavelenghts, dtype=torch.float32).to(device) * 1000000

    return M, theta, wavelengths

def compute_loss(reflective_props: ReflectiveProps, coating: Coating):
    refs = reflective_props.get_properties()

    thicknesses = coating.get_thicknesses()
    M, theta, wavelengths = create_args("../mirror_transformer/" + materials_file, reflective_props)
    preds = tmm('p', M, (thicknesses), theta, wavelengths, device = device)['R']

    lower_bound, upper_bound = torch.chunk(refs, 2, dim=1)
    lower_bound = lower_bound.reshape(reflective_props.get_steps()).to(device)
    upper_bound = upper_bound.reshape(reflective_props.get_steps()).to(device)

    upper_error = torch.clamp(preds - upper_bound, 0, 1)
    lower_error = torch.clamp(lower_bound - preds, 0, 1)

    total_error = torch.sum(upper_error + lower_error)

    return total_error
