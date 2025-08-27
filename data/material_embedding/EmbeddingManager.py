import torch
import torch.nn as nn
import itertools
import yaml
from torch_pca import PCA
from pickle import load, dump
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
        LOSS_SCALE: Scaling factor for loss function given by maximum absolute value of material coefficients.
        materials_refractive_indices: Refractive indices of materials. Shape: (num_materials, |wavelengths|).
        model: Embedding model.
        SAVEPATH: Pathfor saving / loading embeddings.
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
        self.LOSS_SCALE = torch.cat([m.get_coeffs() for m in self.materials]).abs().max().item()
        self.materials_refractive_indices = torch.stack([m.get_refractive_indices() for m in self.materials])
        # pca_lowrank instead of pca?
        self.pca = PCA(n_components = CM().get('material_embedding.dim'))
        self.SAVEPATH = f'data/material_embedding/embeddings_{self.hash_materials()}.pt'
        self.scale_coeffs = []
        self.load_pca()

        self.embeddings = self.refractive_indices_to_embeddings(self.materials_refractive_indices)

    def load_materials(self) -> list[BaseMaterial]:
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

    def refractive_indices_to_embeddings(self, refractive_indices: torch.Tensor):
        assert len(refractive_indices.shape) == 2 or len(refractive_indices.shape) == 3, \
            f"Refractive indices must be of shape (batch_size, |coating|, |wl|) or (|coating|, |wl|)\nFound shape: {refractive_indices.shape}"
        embeddings = self.pca.transform(refractive_indices)
        if len(self.scale_coeffs) == 0:
            # first time running this function
            self.scale_coeffs = [embeddings.min(), embeddings.max() - embeddings.min()]
        embeddings = (embeddings - self.scale_coeffs[0]) / self.scale_coeffs[1]
        return embeddings

    def embeddings_to_refractive_indices(self, embeddings: torch.Tensor):
        assert len(embeddings.shape) == 2 or len(embeddings.shape) == 3, \
            f"Embeddings must be of shape (batch_size, |coating|, embedding_dim) or (|coating|, embedding_dim)\nFound shape: {embeddings.shape}"
        embeddings = embeddings * self.scale_coeffs[1] + self.scale_coeffs[0]
        refractive_indices = self.pca.inverse_transform(embeddings)
        return refractive_indices

    def encode(self, materials: list[BaseMaterial]):
        # TODO: redo with lookup
        """
        Map materials to embeddings.

        Args:
            materials: List of materials.

        Returns:
            Tensor of embeddings. Shape: (batch_size, embedding_dim).
        """
        assert len(materials) > 0, "No materials provided"
        assert isinstance(materials[0], BaseMaterial), "Materials must be of type Material"
        materials_indices = [self.materials_indices[material.get_title()] for material in materials]
        return self.embeddings[materials_indices]

    def decode(self, embedding: torch.Tensor):
        """
        Map embeddings to materials using nearest neighbour.

        Args:
            embedding: Input tensor. Shape: (batch_size, embedding_dim).

        Returns:
            List of materials.
        """
        distances = torch.cdist(embedding, self.embeddings)
        indices = torch.argmin(distances, dim = -1)
        materials = [[self.materials[index] for index in batch] for batch in indices]
        return materials


    def get_refractive_indices_lookup(self):
        return self.materials_refractive_indices

    def get_embeddings_lookup(self):
        return self.embeddings

    def save_pca(self):
        """Save embeddings model."""
        with open(self.SAVEPATH, 'wb') as f:
            dump(self.pca, f, protocol = 5)

    def load_pca(self):
        """Load embeddings model from file or train if file not found."""
        try:
            with open(self.SAVEPATH, 'rb') as f:
                self.pca = load(f)
                self.pca.to(CM().get('device'))
        except FileNotFoundError:
            print(f"Saved embeddings not found. Performing PCA.")
            self.pca.fit(self.materials_refractive_indices)
            self.save_pca()

    def get_materials(self):
        """Return list of materials."""
        return self.materials

    def get_material(self, title: str):
        """Return material with given title."""
        try:
            return list(filter(lambda m: m.get_title() == title, self.materials))[0]
        except IndexError:
            raise ValueError(f"Material with title {title} not found")

    def __str__(self):
        """Return string representation of EmbeddingManager instance."""
        # format by longest material title
        max_title_length = max(len(material.get_title()) for material in self.materials)
        material_embeddings = ''
        for material in self.materials:
            embedding = self.embeddings.cpu().detach().numpy()
            material_embeddings += f"{material.get_title().ljust(max_title_length)}: {embedding}\n"
        return f"Embedding Manager with {len(self.materials)} materials:\n{material_embeddings}"