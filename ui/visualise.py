import torch
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
import sys
sys.path.append(sys.path[0] + '/..')
from data.values.ReflectivePropsPattern import ReflectivePropsPattern
from data.values.ReflectivePropsValue import ReflectivePropsValue
from config import wavelengths


def visualise_spline(preds: ReflectivePropsValue = None, refs: ReflectivePropsPattern = None, filename: str = "visualisation"):
    global wavelengths
    plt.clf()

    wavelengths_cpu = wavelengths.to("cpu")

    wavelengths_spline = torch.linspace(wavelengths_cpu[0], wavelengths_cpu[-1], 1000)

    if refs:
        refs = refs.to("cpu")

        lower_bound = refs.get_lower_bound().detach().cpu()
        wl_lb_spline = make_interp_spline(wavelengths_cpu, lower_bound)
        lower_bound_spline = np.clip(wl_lb_spline(wavelengths_spline), 0, 1)

        upper_bound = refs.get_upper_bound().detach().cpu()
        wl_ub_spline = make_interp_spline(wavelengths_cpu, upper_bound)
        upper_bound_spline = np.clip(wl_ub_spline(wavelengths_spline), 0, 1)
        plt.plot(wavelengths_spline, lower_bound_spline, color='#8ED973')
        plt.plot(wavelengths_spline, upper_bound_spline, color='#C04F15')

    if preds:
        preds = preds.to("cpu")
        value = preds.get_value().detach().cpu()
        wl_vl_spline = make_interp_spline(wavelengths_cpu, value)
        value_spline = np.clip(wl_vl_spline(wavelengths_spline), 0, 1)
        plt.plot(wavelengths_spline, value_spline, color='#D86ECC')

    plt.ylim(0, 1.1)
    plt.savefig(f"out/{filename}.png")

def visualise(preds: ReflectivePropsValue = None, refs: ReflectivePropsPattern = None, filename: str = "visualisation"):
    global wavelengths
    plt.clf()

    wavelengths_cpu = wavelengths.to("cpu")


    if refs:
        refs = refs.to("cpu")
        plt.plot(wavelengths_cpu, refs.get_lower_bound().detach().cpu(), color='#8ED973')
        plt.plot(wavelengths_cpu, refs.get_upper_bound().detach().cpu(), color='#C04F15')

    if preds:
        preds = preds.to("cpu")
        plt.plot(wavelengths_cpu, preds.get_value().detach().cpu(), color='#D86ECC')

    plt.ylim(0, 1.1)
    plt.savefig(f"out/{filename}.png")
