import torch
import sys
sys.path.append(sys.path[0] + '/..')
from data.values.BaseMaterial import BaseMaterial
from data.material_embedding.EmbeddingManager import EmbeddingManager as EM
from utils.ConfigManager import ConfigManager as CM


class Coating():
    """
    Coating class to model thin films.

    Attributes:
        num_layers: Number of layers in coating. Includes substrate and air.
        material_indices: Tensor representing materials encodings. Shape: (batch_size, |coating|, embedding_dim).
        thicknesses: Tensor representing layer thicknesses. Shape: (batch_size, |coating|).
        em: Embedding Manager

    Methods:
        get_encoding: Return encoding of Coating object.
        get_thicknesses: Return thicknesses.
        get_material_indices: Return material encodings.
        get_refractive_indices: Return tensor representing refractive indices for object materials.
        get_materials: Return list of materials.
        get_batch: Return batch at specified index.
        to: Move encodings to device.
        get_device: Return device of Coating object.
    """
    def __init__(self, encoding):
        """
        Initialise Coating instance. Extract material encodings and thicknesses from input tensor and save in object attributes.

        Args:
             encoding: Concatenated tensor of thicknesses and material encodings. Shape: (batch_size, |coating|, embedding_dim + 1)
                       for one batch for one layer: encoding[0] = thickness, encoding[1:] = material encodings
        """
        assert len(encoding.shape) == 3, f"Encoding must have 3 dimensions, found {len(encoding.shape)}"
        assert encoding.shape[2] == 2, f"Final dimension of encoding must be 2, found {encoding.shape[2]}"
        self.num_layers = encoding.shape[1]
        self.material_indices = encoding[:, :, 1].to(torch.int16)
        self.thicknesses = encoding[:, :, 0]

    def get_encoding(self):
        """Return encoding of Coating object."""
        result = torch.stack([self.thicknesses, self.material_indices], dim = -1)
        return result

    def get_thicknesses(self):
        """Return thicknesses."""
        return self.thicknesses

    def get_material_indices(self):
        """Return material encodings."""
        return self.material_indices

    def get_refractive_indices(self):
        """
        Return tensor representing refractive indices for object materials.

        Returns:
            Refractive indices tensor. Shape: (batch_size, |coating|, |wl|)
        """
        return EM().get_refractive_indices(self.material_indices)

    def get_materials(self):
        """Return list of materials."""
        return EM().indices_to_materials(self.material_indices)

    def get_batch(self, index: int):
        """Return batch at specified index."""
        assert index < len(self.thicknesses)
        return Coating(torch.stack([self.thicknesses[index][None], self.material_indices[index][None]], dim = -1))

    # TODO: implement __getitem__

    def __str__(self):
        """Return string representation of object."""
        materials = self.get_materials()
        # format by longest title
        max_title_length = max(len(material.get_title()) for batch in materials for material in batch)
        layers = ''
        for i in range(len(materials)):
            for j in range(len(materials[i])):
                layers += f"{materials[i][j].get_title().ljust(max_title_length)}: {self.thicknesses[i, j].item()}\n"
            layers += '-------\n'
        return f"Coating with {len(materials)} {'batches' if len(materials) > 1 else 'batch'} of {self.num_layers} layers:\n{layers}"

    def __eq__(self, other):
        """
        Compare this Coating object with other object.

        Args:
            other: Object with which to compare.

        Returns:
            True if other is Coating object with same material encodings and thicknesses.
        """
        if isinstance(other, Coating):
            return torch.equal(self.get_thicknesses(), other.get_thicknesses()) and torch.equal(self.get_material_indices(), other.get_material_indices())
        return False

    def to(self, device: str):
        """Move encodings to device."""
        return Coating(self.get_encoding().to(device))

    def get_device(self):
        """Return device of Coating object."""
        return self.thicknesses.device