import pandas as pd
import torch
import os
import sys
sys.path.append(sys.path[0] + '/..')
from data.values.BaseMaterial import BaseMaterial
from utils.ConfigManager import ConfigManager as CM
from inverse_distance_weighting import idw

class MeasuredMaterial(BaseMaterial):
    def __init__(self, title: str, filename: str):
        ownpath = os.path.realpath(__file__)
        material_path = os.path.join(os.path.dirname(os.path.dirname(ownpath)), 'data_files', 'material_measurements', filename)
        assert os.path.exists(material_path), f"No such file: {filename}"
        super().__init__(title.replace('M_', ''))
        self.material_path = material_path

        data = pd.read_csv(self.material_path)
        self.measured_wavelengths = torch.tensor(data['wl'])[:, None]
        self.measured_refractive_indices = torch.tensor(data['n'])[:, None]
        assert not self.measured_wavelengths.isnan().any()
        assert not self.measured_wavelengths.isinf().any()
        assert not self.measured_refractive_indices.isnan().any()
        assert not self.measured_refractive_indices.isinf().any()

        self.idw_tree = idw.tree(self.measured_wavelengths, self.measured_refractive_indices)

    def get_refractive_indices(self):
        wavelengths = CM().get('wavelengths')[:, None].detach().cpu().numpy()
        refractive_indices = self.idw_tree(wavelengths)
        return torch.tensor(refractive_indices).to(CM().get('device')).to(torch.float)

    def get_coeffs(self):
        ...

    def __str__(self):
        """Return string representation of object. Must be implemented by subclasses."""
        return self.title