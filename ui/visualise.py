import torch
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
import sys
sys.path.append(sys.path[0] + '/..')
from data.values.ReflectivityPattern import ReflectivityPattern
from data.values.ReflectivityValue import ReflectivityValue
from utils.ConfigManager import ConfigManager as CM


def visualise_spline(preds: ReflectivityValue = None, refs: ReflectivityPattern = None, filename: str = "visualisation"):
    """
    Visualise prediction using spline interpolation.

    Args:
        preds: predicted reflectivity value. Optional.
        refs: ground truth reflectivity pattern. Optional.
        filename: filename. Optional.
    """
    assert not preds or len(preds.get_value().shape) == 2
    assert not refs or len(refs.get_lower_bound().shape) == 2
    plt.clf()

    wavelengths_cpu = CM().get('wavelengths').to("cpu")

    # linspace of 1000 steps for smooth visualisation
    wavelengths_spline = torch.linspace(wavelengths_cpu[0], wavelengths_cpu[-1], 1000)

    if refs:
        # if refs specified, convert to spline and plot
        refs = refs.to("cpu")

        lower_bound = refs.get_lower_bound()[0].detach().cpu()
        wl_lb_spline = make_interp_spline(wavelengths_cpu, lower_bound)
        lower_bound_spline = np.clip(wl_lb_spline(wavelengths_spline), 0, 1)

        upper_bound = refs.get_upper_bound()[0].detach().cpu()
        wl_ub_spline = make_interp_spline(wavelengths_cpu, upper_bound)
        upper_bound_spline = np.clip(wl_ub_spline(wavelengths_spline), 0, 1)
        plt.plot(wavelengths_spline, lower_bound_spline, color='#8ED973')
        plt.plot(wavelengths_spline, upper_bound_spline, color='#C04F15')

    if preds:
        # if preds specified, convert to spline and plot
        preds = preds.to("cpu")
        value = preds.get_value()[0].detach().cpu()
        wl_vl_spline = make_interp_spline(wavelengths_cpu, value)
        value_spline = np.clip(wl_vl_spline(wavelengths_spline), 0, 1)
        plt.plot(wavelengths_spline, value_spline, color='#D86ECC')

    plt.ylim(0, 1.1)
    plt.savefig(f"out/{filename}.png")

def visualise(preds: ReflectivityValue = None, refs: ReflectivityPattern = None, filename: str = "visualisation"):
    """
    Visualise prediction using linear interpolation.

    Args:
        preds: predicted reflectivity value. Optional.
        refs: ground truth reflectivity pattern. Optional.
        filename: filename. Optional.
    """
    assert not preds or len(preds.get_value().shape) == 2
    assert not refs or len(refs.get_lower_bound().shape) == 2
    plt.clf()

    wavelengths_cpu = CM().get('wavelengths').to("cpu")

    if refs:
        # plot refs if specified
        refs = refs.to("cpu")
        plt.plot(wavelengths_cpu, refs.get_lower_bound()[0].detach().cpu(), color='#8ED973')
        plt.plot(wavelengths_cpu, refs.get_upper_bound()[0].detach().cpu(), color='#C04F15')

    if preds:
        # plot preds if specified
        preds = preds.to("cpu")
        plt.plot(wavelengths_cpu, preds.get_value()[0].detach().cpu(), color='#D86ECC')

    plt.ylim(0, 1.1)
    plt.savefig(f"out/{filename}.png")
