import torch
import torch.nn.functional as F
import yaml
import sys

sys.path.append(sys.path[0] + '/..')
from data.values.BaseMaterial import BaseMaterial
from data.values.SellmeierMaterial import SellmeierMaterial
from data.values.ConstantRIMaterial import ConstantRIMaterial
from utils.ConfigManager import ConfigManager as CM
from utils.os_utils import short_hash

class EmbeddingManager:
    """
    Embedding Manager class for mapping materials to embeddings.

    Materials are characterised by single-value refractive index (type ConstantRIMaterial) or multiple-value Sellmeier coefficients (type SellmeierMaterial).
    Real materials are few, therefore the multidimensional space of Sellmeier coefficients is sparse.
    This can unnecessarily complicate the optimisation landscape, so materials are embedded into a lower-dimensional space.
    Embedding space size is defined in config.material_embedding.dim.

    This class is a singleton.

    Attributes:
        materials: List of materials.
        num_materials: Number of materials.
        refractive_indices: Refractive indices of materials. Shape: (num_materials, |wavelengths|).
        model: Embedding model.
        SAVEPATH: Path for saving / loading embeddings.
        embeddings: Embeddings model parameter. Shape: (num_materials, embedding_dim).

    Methods:

    """
    _instance = None

    def __new__(cls):
        """Create singleton instance of EmbeddingManager class."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialise EmbeddingManager instance."""
        self.materials = []
        self.materials_indices = dict()
        self.load_materials()
        self.num_materials = len(self.materials)
        self.refractive_indices = torch.stack([m.get_refractive_indices() for m in self.materials])

    def get_material_indices(self, materials):
        return torch.tensor([self.materials_indices[m] for m in materials])

    def load_materials(self):
        # TODO: map benutzen und filter wenn nötig (wie in CM)
        """
        Load materials from data file.

        Material type detection based on provided coefficients (B and C for SellmeierMaterial or R for ConstantRIMaterial).
        All loaded materials must have same type.

        Raises:
            ValueError: If no thin films are specified in config.
            AssertionError: If loaded materials have different types.

        Returns:
            List of materials.
        """
        with open(CM().get('material_embedding.data_file'), 'r') as file:
            data = yaml.safe_load(file)
        thin_films = CM().get('materials.thin_films')
        if len(thin_films) == 0:
            raise ValueError("No thin films specified")

        materials = []
        for m in data['materials']:
            if m['title'] not in thin_films:
                continue
            if 'B' in m and 'C' in m:
                # material has Sellmeier coefficients B and C
                materials.append(SellmeierMaterial(m['title'], m['B'], m['C']))
            elif 'R' in m:
                # material has single-value refractive index
                materials.append(ConstantRIMaterial(m['title'], m['R']))
            else:
                raise ValueError(f"Material {m['title']} must have either 'B' and 'C' or 'R'")
        # sort so that all features have the same material order
        materials.sort()
        # prepend substrate and air
        for m in data['materials']:
            if m['title'] == CM().get('materials.substrate'):
                if 'B' in m and 'C' in m:
                    substrate = SellmeierMaterial(m['title'], m['B'], m['C'])
                elif 'R' in m:
                    substrate = ConstantRIMaterial(m['title'], m['R'])
                else:
                    raise ValueError(f"Material {m['title']} must have either 'B' and 'C' or 'R'")
            if m['title'] == CM().get('materials.air'):
                if 'B' in m and 'C' in m:
                    air = SellmeierMaterial(m['title'], m['B'], m['C'])
                elif 'R' in m:
                    air = ConstantRIMaterial(m['title'], m['R'])
                else:
                    raise ValueError(f"Material {m['title']} must have either 'B' and 'C' or 'R'")
        materials = [substrate, air] + materials

        # ensure all materials have same type
        material_type = type(materials[0])
        for m in materials:
            assert isinstance(m, material_type), f"All materials must be of same type."

        self.materials = materials
        self.materials_indices = {materials[i].get_title(): i for i in range(len(materials))}

    def hash_materials(self):
        """Hash materials to use as filename for saving / loading embeddings."""
        # does this need more information?
        material_hashes = [hash(material) for material in self.materials]
        return short_hash(tuple(material_hashes))

    def get_refractive_indices(self, material_indices: torch.Tensor):
        # this needs to be differentiable
        # input of shape (batch, seq_len, 1)
        assert len(material_indices.shape) == 2
        assert material_indices.dtype in [torch.int8, torch.int16, torch.int32, torch.int64], "Indices must be integer values"
        long_indices = material_indices.to(torch.long)
        mask = F.one_hot(long_indices, len(self.refractive_indices)).to(torch.float)
        return mask @ self.refractive_indices

    def indices_to_materials(self, material_indices: torch.Tensor):
        materials = []
        for sequence in material_indices:
            seq_materials = [self.materials[i] for i in sequence]
            materials.append(seq_materials)
        return materials

    def materials_to_indices(self, materials: list[list[BaseMaterial]]):
        # bidimensional list
        assert len(materials) > 0, "No materials provided"
        return torch.tensor([[self.materials_indices[m.get_title()] for m in sequence] for sequence in materials], device = CM().get('device'))

    def get_refractive_indices_table(self):
        return self.refractive_indices


    def get_materials(self):
        """Return list of materials."""
        return self.materials

    def get_material_by_title(self, title: str):
        """Return material with given title."""
        try:
            return list(filter(lambda m: m.get_title() == title, self.materials))[0]
        except IndexError:
            raise ValueError(f"Material with title {title} not found")

    def __str__(self):
        """Return string representation of EmbeddingManager instance."""
        return f"Embedding Manager with {len(self.materials)} materials:\n{self.materials}"