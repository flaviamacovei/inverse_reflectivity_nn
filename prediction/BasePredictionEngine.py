from abc import ABC, abstractmethod
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import sys
sys.path.append(sys.path[0] + '/..')
from data.values.ReflectivePropsPattern import ReflectivePropsPattern
from data.values.ReflectivePropsValue import ReflectivePropsValue

class BasePredictionEngine(ABC):

    def __init__(self):
        pass
    @abstractmethod
    def predict(self, pattern: ReflectivePropsPattern):
        pass

    def visualise(self, refs: ReflectivePropsPattern, preds: ReflectivePropsValue, epoch):
        plt.clf()

        start_wl = refs.get_start_wl()
        end_wl = refs.get_end_wl()
        steps = refs.get_lower_bound().shape[0]

        wavelengths = torch.linspace(start_wl, end_wl, steps)
        wavelengths_spline = torch.linspace(start_wl, end_wl, 1000)

        lower_bound = refs.get_lower_bound().detach().cpu()
        wl_lb_spline = make_interp_spline(wavelengths, lower_bound)
        lower_bound_spline = wl_lb_spline(wavelengths_spline)

        upper_bound = refs.get_upper_bound().detach().cpu()
        wl_ub_spline = make_interp_spline(wavelengths, upper_bound)
        upper_bound_spline = wl_ub_spline(wavelengths_spline)

        value = preds.get_value().detach().cpu()
        wl_vl_spline = make_interp_spline(wavelengths, value)
        value_spline = wl_vl_spline(wavelengths_spline)

        plt.plot(wavelengths_spline, value_spline, color = '#D86ECC')
        plt.plot(wavelengths_spline, lower_bound_spline, color = '#8ED973')
        plt.plot(wavelengths_spline, upper_bound_spline, color = '#C04F15')
        plt.ylim(0, 1.2)
        plt.savefig(f"out/graph_{epoch}.png")