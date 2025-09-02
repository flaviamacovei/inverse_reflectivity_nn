import torch
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
import pandas as pd
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
    assert not preds or len(preds.get_value().shape) == 2, f"preds shape must have length 2 found {len(preds.get_value().shape)}"
    assert not refs or len(refs.get_lower_bound().shape) == 2, f"refs shape must have length 2 found {len(refs.get_lower_bound().shape)}"
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
    assert not preds or len(
        preds.get_value().shape) == 2, f"preds shape must have length 2 found {len(preds.get_value().shape)}"
    assert not refs or len(
        refs.get_lower_bound().shape) == 2, f"refs shape must have length 2 found {len(refs.get_lower_bound().shape)}"
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

def visualise_errors(errors: pd.DataFrame, filename: str = "errors_graph", log_scale: bool = False):
    sorted = errors.sort_values(by='median').reset_index(drop=True)
    boxplot_data = []
    for _, row in sorted.iterrows():
        box = {
            'med': row['median'],
            'mean': row['mean'],
            'q1': row['q_25'],
            'q3': row['q_75'],
            'whislo': row['min'],
            'whishi': row['max'],
        }
        boxplot_data.append(box)
    fig, ax = plt.subplots(figsize = (1.2 * len(sorted), 6))
    ax.bxp(boxplot_data, showfliers = False, showmeans = True)
    if log_scale:
        ax.set_yscale('log')
    ax.set_xticks(np.arange(1, len(sorted) + 1))
    ax.set_xticklabels(sorted['model'], rotation=45, ha='right')
    y_max = sorted['max'].max()
    ax.set_ylim(y_max * -0.05, y_max * 1.05)  # epsilon = 5% of max
    ax.set_xlabel('Model')
    ax.set_ylabel('Error')
    ax.set_title('Models Prediction Error')
    plt.tight_layout()
    plt.savefig(f"out/{filename}.png")
