import numpy as np

from utils.ObservableList import ObservableList
import torch
import sys
sys.path.append(sys.path[0] + '/..')
from data.values.Region import Region
from data.values.ReflectivePropsPattern import ReflectivePropsPattern
from utils.ConfigManager import ConfigManager as CM
from utils.tmm_utils import get_wavelength_index

class UserInput():
    def __init__(self):
        self.regions = ObservableList()
        self.regions.append_action = self.append_action

    def append_action(self, region: Region):
        assert region, "Region must not be None."
        assert CM().get('wavelengths')[0] <= region.get_start_wl() and CM().get('wavelengths')[-1] >= region.get_end_wl(), "Region must be within frame."
        assert isinstance(get_wavelength_index(region.get_start_wl()), int) and isinstance(get_wavelength_index(region.get_end_wl()), int), "Region must be interval of frame."
        for existing_region in self.regions:
            if not (
                    region.get_start_wl() >= existing_region.get_end_wl() or region.get_end_wl() <= existing_region.get_start_wl()):
                raise AssertionError("New region overlaps with an existing region.")

    def read_int_from_input(self, prompt):
        user_input = input(prompt)
        try:
            return int(user_input)
        except ValueError:
            print("Invalid input. Please enter an integer.")
            return self.read_int_from_input(prompt)

    def read_float_from_input(self, prompt):
        user_input = input(prompt)
        try:
            return float(user_input)
        except ValueError:
            print("Invalid input. Please enter a float.")
            return self.read_float_from_input(prompt)


    def read_regions(self):
        print("Please specify regions by start wavelength (in nm), end wavelength (in nm), and target reflectivity.")
        cont = True
        while cont:
            region_start_idx = self.read_int_from_input("Start wavelength: ") * 1.e-3
            region_end_idx = self.read_int_from_input("End wavelength: ") * 1.e-3
            region_value = self.read_float_from_input("Target reflectivity: ")
            try:
                region = Region(region_start_idx, region_end_idx, region_value)
                self.regions.append(region)
            except AssertionError as e:
                print(f"Provided region is invalid: {e}\nPlease try again.\n")
            cont = input("Specify another region? (y/n): ") == "y"

        print("\nSpecified regions:")
        for region in self.regions:
            print(region)
        if input("Confirm? (y/n): ") != "y":
            self.regions = ObservableList()
            return self.read_regions()
        print("")


    def run(self):
        self.read_regions()

    def to_reflective_props_pattern(self):
        lower_bound = torch.zeros((1, CM().get('wavelengths').shape[0]), device = CM().get('device'))
        upper_bound = torch.ones((1, CM().get('wavelengths').shape[0]), device = CM().get('device'))

        for region in self.regions:
            lower_bound[:, get_wavelength_index(region.get_start_wl()):get_wavelength_index(region.get_end_wl())] = region.get_value()
            upper_bound[:, get_wavelength_index(region.get_start_wl()):get_wavelength_index(region.get_end_wl())] = region.get_value()

        lower_bound = torch.clamp(lower_bound - CM().get('tolerance') / 2, 0, 1)
        upper_bound = torch.clamp(upper_bound + CM().get('tolerance') / 2, 0, 1)

        return ReflectivePropsPattern(lower_bound, upper_bound)
