import numpy as np
import torch
from tmm_fast import coh_tmm as tmm
import sys
sys.path.append(sys.path[0] + '/..')
from data.values.ReflectiveProps import ReflectiveProps
from data.values.Coating import Coating
from mirror_transformer.direct_opt import SM, Material, Stack

class LossCalculator():
    def __init__(self, materials_file: str, reflective_props: ReflectiveProps, num_stacks: int, device: str, pol: str):
        self.stack = self.make_stack(materials_file)
        self.reflective_props = reflective_props
        self.num_stacks = num_stacks
        self.device = device

        self.pol = pol
        wavelenghts = np.linspace(self.reflective_props.get_start_wl(), self.reflective_props.get_end_wl(), self.reflective_props.get_steps()) * (10 ** (-9))
        self.M = torch.tensor(self.stack.make_M(wavelenghts, self.num_stacks), dtype = torch.float32).to(device)
        self.theta = torch.tensor(np.linspace(0, 0, 1) * (np.pi / 180), dtype = torch.float32).to(self.device)

        self.wavelengths = torch.tensor(wavelenghts, dtype=torch.float32).to(device) * 1000000

    def make_stack(self, filepath):
        sm = SM()
        with open(filepath, "r", encoding="ISO-8859-1") as f:
            for line in f:
                sm.run(line)
        materials = [Material(m) for m in sm.materials]
        st = Stack(sm.front, materials)
        return st

    def compute_loss(self, coating: Coating):
        refs = self.reflective_props.get_properties()

        thicknesses = coating.get_thicknesses()
        preds = tmm(self.pol, self.M, (thicknesses), self.theta, self.wavelengths, device = self.device)['R']

        print(f"preds: {preds}")

        lower_bound, upper_bound = torch.chunk(refs, 2, dim=1)
        lower_bound = lower_bound.reshape(self.reflective_props.get_steps()).to(self.device)
        upper_bound = upper_bound.reshape(self.reflective_props.get_steps()).to(self.device)

        upper_error = torch.clamp(preds - upper_bound, 0, 1)
        lower_error = torch.clamp(lower_bound - preds, 0, 1)

        total_error = torch.sum(upper_error + lower_error)

        return total_error
