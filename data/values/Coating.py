import torch
import sys
sys.path.append(sys.path[0] + '/..')
from data.values.Material import Material
from data.material_embedding.EmbeddingManager import EmbeddingManager
from utils.ConfigManager import ConfigManager as CM


class Coating():


    def __init__(self, materials: list [list[Material]], thicknesses: torch.Tensor):
        assert len(thicknesses.shape) == 2
        assert len(materials) == thicknesses.shape[0], f"Materials and thicknesses must have same batch size, got materials: {len(materials)} and thicknesses: {thicknesses.shape[0]}" # batch size
        assert len(materials[0]) == thicknesses.shape[1], f"Materials and thicknesses must have same number of layers, got materials: {len(materials[0])} and thicknesses: {thicknesses.shape[1]}" # number of layers
        self.num_layers = thicknesses.shape[1]
        self.layers = [[(material, thickness) for material, thickness in zip(materials[i], thicknesses[i])] for i in range(thicknesses.shape[0])]
        self.em = EmbeddingManager()

    @classmethod
    def from_encoding(cls, encoding: torch.Tensor):
        assert len(encoding.shape) == 3
        assert encoding.shape[2] == CM().get('material_embedding.dim') + 1
        thicknesses = encoding[:, :, 0]
        material_encoding = encoding[:, :, 1:]
        materials = [EmbeddingManager().decode(material_encoding[i])[0] for i in range(material_encoding.shape[0])]
        return cls(materials, thicknesses)


    def get_layers(self):
        return self.layers

    def get_encoding(self):
        encoding = torch.zeros((len(self.layers), self.num_layers, CM().get('material_embedding.dim') + 1), device = CM().get('device'))
        for i, item in enumerate(self.layers):
            encoding[i, :, 0] = torch.stack([layer[1] for layer in item])
            encoding[i, :, 1:] = self.em.encode([layer[0] for layer in item])
        return encoding

    def get_thicknesses(self):
        return torch.stack([torch.cat([layer[1].view(1) for layer in item]) for item in self.layers])

    def get_refractive_indices(self):
        refractive_indices = torch.zeros((len(self.layers), self.num_layers, CM().get('wavelengths').shape[0]), device = CM().get('device'))
        for i, item in enumerate(self.layers):
            refractive_indices[i] = torch.stack([layer[0].get_refractive_indices() for layer in item])
        return refractive_indices

    def get_materials(self):
        return [[layer[0] for layer in item] for item in self.layers]

    def get_batch(self, index: int):
        assert index < len(self.layers)
        materials = [self.layers[index][i][0] for i in range(self.num_layers)]
        thicknesses = [self.layers[index][i][1] for i in range(self.num_layers)]
        return Coating(materials, thicknesses)

    def __str__(self):
        max_title_length = max(len(layer[0].get_title()) for item in self.layers for layer in item)
        layers = ''
        for item in self.layers:
            for layer in item:
                layers += f"{layer[0].get_title().ljust(max_title_length)}: {layer[1]}\n"
                if layer[0].get_title() == CM().get('materials.air'):
                    print(f"continuing")
                    break
            layers += '-------\n'
        return f"Coating with {len(self.layers)} {'batches' if len(self.layers) > 1 else 'batch'} of {self.num_layers} layers:\n{layers}"

    def __eq__(self, other):
        if isinstance(other, Coating):
            return self.layers == other.get_layers()
        return False