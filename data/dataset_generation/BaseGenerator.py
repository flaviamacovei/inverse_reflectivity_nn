from abc import ABC, abstractmethod
import random
import torch
import sys
sys.path.append(sys.path[0] + '/../..')
from data.values.Coating import Coating
from utils.ConfigManager import ConfigManager as CM
from data.material_embedding.EmbeddingManager import EmbeddingManager as EM

class BaseGenerator(ABC):
    def __init__(self, num_points: int = 1):
        self.num_points = num_points
        self.TOLERANCE = CM().get('tolerance')

    def make_random_coating(self):
        num_layers = random.randint(CM().get('layers.min'), CM().get('layers.max'))

        substrate = EM().get_material(CM().get('materials.substrate'))
        air = EM().get_material(CM().get('materials.air'))
        thin_film_materials = list(filter(lambda x: x != substrate and x != air, EM().get_materials()))
        thin_films = random.choices(thin_film_materials, k = num_layers)

        coating_materials = [substrate] + thin_films + [air]
        coating_materials.extend([air] * (CM().get('layers.max') - num_layers))

        thicknesses = torch.zeros((1, CM().get('layers.max') + 2), device = CM().get('device'))
        thicknesses[0, :num_layers] = torch.rand((1, num_layers), device = CM().get('device')) * 0.2

        encoding = torch.cat([thicknesses[:, :, None], EM().encode(coating_materials)[None]], dim = 2)
        return Coating(encoding)

    @abstractmethod
    def make_point(self):
        pass

    def generate(self):
        points = []
        for i in range(self.num_points):
            if i % (max(self.num_points // 10, 1)) == 0:
                print(f"{i}/{self.num_points}")
            points.append(self.make_point())
            torch.cuda.empty_cache()
        return points
