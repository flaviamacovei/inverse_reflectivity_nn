import torch
import sys
sys.path.append(sys.path[0] + '/..')
from data.values.Material import Material
from data.material_embedding.EmbeddingManager import EmbeddingManager
from utils.ConfigManager import ConfigManager as CM


class Coating():

    def __init__(self, encoding):
        assert len(encoding.shape) == 3
        assert encoding.shape[2] == CM().get('material_embedding.dim') + 1
        self.num_layers = encoding.shape[1]
        self.material_encodings = encoding[:, :, 1:]
        self.thicknesses = encoding[:, :, 0]
        self.em = EmbeddingManager()

    def get_encoding(self):
        result = torch.cat([self.thicknesses[:, :, None], self.material_encodings], dim = 2)
        return result

    def get_thicknesses(self):
        return self.thicknesses

    def get_material_encodings(self):
        return self.material_encodings

    def get_refractive_indices(self):
        nearest_neighbours = self.em.get_nearest_neighbours(self.material_encodings)
        refractive_indices = self.em.embedding_to_refractive_indices(nearest_neighbours)
        return refractive_indices

    def get_materials(self):
        return self.em.decode(self.material_encodings)

    def get_batch(self, index: int):
        assert index < len(self.thicknesses)
        return Coating(torch.cat([self.thicknesses[index][None, :, None], self.material_encodings[index][None]], dim = 2))

    # TODO: implement __getitem__

    def __str__(self):
        materials = self.get_materials()
        max_title_length = max(len(material.get_title()) for batch in materials for material in batch)
        layers = ''
        for i in range(len(materials)):
            for j in range(len(materials[i])):
                layers += f"{materials[i][j].get_title().ljust(max_title_length)}: {self.thicknesses[i, j]}\n"
            layers += '-------\n'
        return f"Coating with {len(materials)} {'batches' if len(materials) > 1 else 'batch'} of {self.num_layers} layers:\n{layers}"

    def __eq__(self, other):
        if isinstance(other, Coating):
            return torch.equal(self.get_thicknesses(), other.get_thicknesses()) and torch.equal(self.get_material_encodings(), other.get_material_encodings())
        return False

    def to(self, device: str):
        return Coating(self.get_encoding().to(device))

    def get_device(self):
        return self.thicknesses.device