from abc import ABC, abstractmethod
import random
import torch
import sys
sys.path.append(sys.path[0] + '/../..')
from data.values.Coating import Coating
from data.values.RefractiveIndex import RefractiveIndex
from utils.ConfigManager import ConfigManager as CM
from data.material_embedding.EmbeddingManager import EmbeddingManager as EM

class BaseGenerator(ABC):
    def __init__(self, num_points: int):
        self.num_points = num_points
        self.TOLERANCE = CM().get('tolerance')

    def make_random_coating(self):
        num_layers = random.randint(CM().get('layers.min'), CM().get('layers.max'))

        substrate = EM().get_material(CM().get('materials.substrate'))
        air = EM().get_material(CM().get('materials.air'))
        thin_film_materials = list(filter(lambda x: x != substrate and x != air, EM().get_materials()))
        thin_films = random.choices(thin_film_materials, k = num_layers - 2)

        coating_materials = [[substrate] + thin_films + [air]]

        thicknesses = torch.rand((1, num_layers), device = CM().get('device')) / 1.0e6

        return Coating(coating_materials, thicknesses)

    @abstractmethod
    def make_point(self):
        pass

    def generate(self):
        points = []
        for i in range(self.num_points):
            if i % (max(self.num_points // 10, 1)) == 0:
                print(f"{i}/{self.num_points}")
            points.append(self.make_point())
        return points
