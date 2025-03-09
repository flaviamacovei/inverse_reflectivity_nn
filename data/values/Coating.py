import torch
import sys
sys.path.append(sys.path[0] + '/..')
from data.values.Material import Material
from data.material_embedding.EmbeddingManager import EmbeddingManager
from utils.ConfigManager import ConfigManager as CM


class Coating():

    def __init__(self, materials: list[Material], thicknesses: torch.Tensor):
        assert len(materials) == thicknesses.shape[0]
        self.layers = [(material, thickness) for material, thickness in zip(materials, thicknesses)]
        self.em = EmbeddingManager()

    def get_layers(self):
        return self.layers

    def get_encoding(self):
        encoding = torch.zeros((CM().get('material_embedding.dim') + 1, len(self.layers)), device = CM().get('device'))
        print(f"encoding zeros: {encoding}")
        for i, layer in enumerate(self.layers):
            encoding[0, i] = layer[1]
            encoding[1:, i] = self.em.encode([layer[0]])
        print(f"encoding: {encoding}")

    def __str__(self):
        max_title_length = max(len(layer[0].get_title()) for layer in self.layers)
        layers = ''
        for layer in self.layers:
            layers += f"{layer[0].get_title().ljust(max_title_length)}: {layer[1]}\n"
        return f"Coating with {len(self.layers)} materials:\n{layers}"

    def __eq__(self, other):
        if isinstance(other, Coating):
            return self.layers == other.get_layers()
        return False