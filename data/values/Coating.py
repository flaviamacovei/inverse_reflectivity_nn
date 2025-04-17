import torch
import sys
sys.path.append(sys.path[0] + '/..')
from data.values.Material import Material
from data.material_embedding.EmbeddingManager import EmbeddingManager
from utils.ConfigManager import ConfigManager as CM


class Coating():
    """
    Coating class to model thin films.

    Attributes:
        num_layers: Number of layers in coating. Includes substrate and air.
        material_encodings: Tensor representing materials encodings. Shape: (batch_size, |coating|, embedding_dim).
        thicknesses: Tensor representing layer thicknesses. Shape: (batch_size, |coating|).
        em: Embedding Manager

    Methods:
        get_encoding: Return encoding of Coating object.
        get_thicknesses: Return thicknesses.
        get_material_encodings: Return material encodings.
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
        assert len(encoding.shape) == 3
        assert encoding.shape[2] == CM().get('material_embedding.dim') + 1
        self.num_layers = encoding.shape[1]
        self.material_encodings = encoding[:, :, 1:]
        self.thicknesses = encoding[:, :, 0]
        self.em = EmbeddingManager()

    def get_encoding(self):
        """Return encoding of Coating object."""
        result = torch.cat([self.thicknesses[:, :, None], self.material_encodings], dim = 2)
        return result

    def get_thicknesses(self):
        """Return thicknesses."""
        return self.thicknesses

    def get_material_encodings(self):
        """Return material encodings."""
        return self.material_encodings

    def get_refractive_indices(self):
        """
        Return tensor representing refractive indices for object materials.

        Returns:
            Refractive indices tensor. Shape: (batch_size, |coating|, |config.wavelengths|)
        """
        nearest_neighbours = self.em.get_nearest_neighbours(self.material_encodings)
        refractive_indices = self.em.embedding_to_refractive_indices(nearest_neighbours)
        return refractive_indices

    def get_materials(self):
        """Return list of materials."""
        return self.em.decode(self.material_encodings)

    def get_batch(self, index: int):
        """Return batch at specified index."""
        assert index < len(self.thicknesses)
        return Coating(torch.cat([self.thicknesses[index][None, :, None], self.material_encodings[index][None]], dim = 2))

    # TODO: implement __getitem__

    def __str__(self):
        """Return string representation of object."""
        materials = self.get_materials()
        # format by longest title
        max_title_length = max(len(material.get_title()) for batch in materials for material in batch)
        layers = ''
        for i in range(len(materials)):
            for j in range(len(materials[i])):
                layers += f"{materials[i][j].get_title().ljust(max_title_length)}: {self.thicknesses[i, j]}\n"
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
            return torch.equal(self.get_thicknesses(), other.get_thicknesses()) and torch.equal(self.get_material_encodings(), other.get_material_encodings())
        return False

    def to(self, device: str):
        """Move encodings to device."""
        return Coating(self.get_encoding().to(device))

    def get_device(self):
        """Return device of Coating object."""
        return self.thicknesses.device