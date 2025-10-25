import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
import pandas as pd
import sys
sys.path.append(sys.path[0] + '/..')
from data.values.ReflectivityPattern import ReflectivityPattern
from data.values.ReflectivityValue import ReflectivityValue
from utils.ConfigManager import ConfigManager as CM
from utils.os_utils import get_unique_filename
from utils.math_utils import largest_prime_factor


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
        plt.fill_between(wavelengths_spline, lower_bound_spline,
                         upper_bound_spline, color='#D4D4D4')

    if preds:
        # if preds specified, convert to spline and plot
        preds = preds.to("cpu")
        value = preds.get_value()[0].detach().cpu()
        wl_vl_spline = make_interp_spline(wavelengths_cpu, value)
        value_spline = np.clip(wl_vl_spline(wavelengths_spline), 0, 1)
        plt.plot(wavelengths_spline, value_spline, color='#33638D')

    plt.ylim(-0.01, 1.01)
    plt.savefig(f"out/{filename}.png")
    plt.close()

def visualise(preds: ReflectivityValue = None, refs: ReflectivityPattern = None, filename: str = "visualisation", svg: bool = False, min_x: int = 0, max_x: int = -1):
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
    wavelengths_cpu = wavelengths_cpu[min_x:max_x]

    if preds is not None:
        batch_size = preds.get_batch_size()
    elif refs is not None:
        batch_size = refs.get_batch_size()
    else:
        batch_size = 1

    num_plots = min(6, batch_size)
    n_cols = largest_prime_factor(num_plots)
    n_rows = num_plots // n_cols

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    if num_plots == 1:
        axs = [axs]
    for i, col in enumerate(axs):
        for j, ax in enumerate(col):
            index = i * n_cols + j
            if refs:
                # plot refs if specified
                refs = refs.to("cpu")
                ax.fill_between(wavelengths_cpu, refs.get_lower_bound()[index, min_x:max_x].detach().cpu(), refs.get_upper_bound()[index, min_x:max_x].detach().cpu(), color = '#D4D4D4')
            if preds:
                # plot preds if specified
                preds = preds.to("cpu")
                ax.plot(wavelengths_cpu, preds.get_value()[index, min_x:max_x].detach().cpu(), color='#33638D')
            ax.set_ylim(-0.01, 1.01)

    # plt.ylim(-0.01, 1.01)
    extention = "svg" if svg else "png"
    fig.savefig(f"out/{filename}.{extention}")
    plt.close()

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
    plt.close(fig)

def visualise_matrix(matrix: torch.Tensor, title: str = "visualisation"):
    assert len(matrix.shape) == 2, f"Expected two-dimensional matrix found {len(matrix.shape)}"
    matrix_np = matrix.detach().cpu().numpy()
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.imshow(matrix_np, cmap='viridis', extent = (0, 10, 0, 10), interpolation = 'none')

    # set axes
    y_shape, x_shape = matrix_np.shape
    x_space = ax.get_xticks()
    y_space = ax.get_yticks()
    if len(x_space) > x_shape:
        x_space = np.linspace(x_space[0], x_space[-1], x_shape)
    if len(y_space) > y_shape:
        y_space = np.linspace(y_space[0], y_space[-1], y_shape)
    ax.set_xticks(x_space)
    ax.set_yticks(y_space)
    x_ticklabels = np.linspace(0, x_shape, len(x_space))
    y_ticklabels = np.linspace(0, y_shape, len(y_space))
    ax.set_xticklabels(x_ticklabels)
    ax.set_yticklabels(y_ticklabels)

    filename = get_unique_filename(f"out/{title}.png")
    plt.savefig(filename)
    plt.close()


def model_snapshot(model: nn.Module, filename: str = "model_snapshot"):
    num_layers = 0
    for _, _ in model.named_parameters():
        num_layers += 1
    n_cols = largest_prime_factor(num_layers)
    n_rows = num_layers // n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    if num_layers == 1:
        axs = [axs]
    for i, col in enumerate(axs):
        for j, ax in enumerate(col):
            index = i * n_cols + j
            name, param = list(model.named_parameters())[index]
            if len(param.shape) == 1:
                param = param[None]
                ax.imshow(param.detach().cpu().numpy(), cmap='viridis', extent=(0, 10, 0, 10), interpolation='none')
                ax.set_title(name.split('.')[-1])
                y_shape, x_shape = param.shape
                x_space = ax.get_xticks()
                y_space = ax.get_yticks()
                if len(x_space) > x_shape:
                    x_space = np.linspace(x_space[0], x_space[-1], x_shape)
                if len(y_space) > y_shape:
                    y_space = np.linspace(y_space[0], y_space[-1], y_shape)
                ax.set_xticks(x_space)
                ax.set_yticks(y_space)
                x_ticklabels = np.linspace(0, x_shape, len(x_space))
                y_ticklabels = np.linspace(0, y_shape, len(y_space))
                ax.set_xticklabels(x_ticklabels)
                ax.set_yticklabels(y_ticklabels)
    fig.savefig(f"out/{filename}.png")
    plt.close()