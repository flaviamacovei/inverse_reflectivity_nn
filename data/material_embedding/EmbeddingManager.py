import torch
import torch.nn as nn
import itertools
import yaml
import sys
sys.path.append(sys.path[0] + '/..')
from data.values.Material import Material
from data.values.SellmeierMaterial import SellmeierMaterial
from data.values.ConstantRIMaterial import ConstantRIMaterial
from utils.ConfigManager import ConfigManager as CM
from utils.os_utils import short_hash

class EmbeddingModel(nn.Module):
    def __init__(self, in_dim, num_materials, embedding_dim, fixed_points = None):
        super().__init__()
        hidden_dim = 4
        self.embeddings = nn.Parameter(torch.randn(num_materials, embedding_dim))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        self.fixed_points = fixed_points

    def forward(self, x):
        keys = torch.stack(list(self.fixed_points.keys())).repeat(x.shape[0], 1, 1)
        mask = (x[:, None] == keys).all(dim=-1)
        output = torch.empty((x.shape[0], CM().get('material_embedding.dim')), device=CM().get('device'))
        fixed_points_values = torch.stack(list(self.fixed_points.values()))
        output[mask.any(dim=-1)] = fixed_points_values.repeat(x.shape[0], 1, 1)[mask]
        output[~mask.any(dim=-1)] = self.net(x[~mask.any(dim=-1)])
        return output

    def get_embeddings(self):
        return self.embeddings

    def set_embeddings(self, embeddings):
        self.embeddings = embeddings

class EmbeddingManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.materials = self.load_materials()
        self.num_materials = len(self.materials)
        self.LOSS_SCALE = torch.cat([m.get_coeffs() for m in self.materials]).abs().max().item()
        self.materials_refractive_indices = torch.stack([m.get_refractive_indices() for m in self.materials])

        indim = self.materials[0].get_coeffs().shape[0]

        substrate = self.get_material(CM().get('materials.substrate')).get_coeffs()
        air = self.get_material(CM().get('materials.air')).get_coeffs()
        fixed_points = {
            substrate: torch.ones((CM().get('material_embedding.dim')), device = CM().get('device')),
            air: torch.zeros((CM().get('material_embedding.dim')), device = CM().get('device'))
        }
        self.model = EmbeddingModel(indim, self.num_materials, CM().get('material_embedding.dim'), fixed_points).to(CM().get('device'))

        self.SAVEPATH = f'data/material_embedding/embeddings_{self.hash_materials()}.pt'
        self.load_embeddings()
        self.embeddings = self.model(torch.stack([m.get_coeffs() for m in self.materials]))

    def load_materials(self) -> list[Material]:
        with open(CM().get('material_embedding.data_file'), 'r') as file:
            data = yaml.safe_load(file)
        thin_films = CM().get('materials.thin_films')
        if len(thin_films) == 0:
            raise ValueError("No thin films specified")

        allowed_titles = thin_films + [CM().get('materials.substrate'), CM().get('materials.air')]
        materials = []
        for m in data['materials']:
            if m['title'] not in allowed_titles:
                continue
            if 'B' in m and 'C' in m:
                materials.append(SellmeierMaterial(m['title'], m['B'], m['C']))
            elif 'R' in m:
                materials.append(ConstantRIMaterial(m['title'], m['R']))
            else:
                raise ValueError(f"Material {m['title']} must have either 'B' and 'C' or 'R'")

        material_type = type(materials[0])
        for m in materials:
            assert isinstance(m, material_type), f"All materials must be of same type."
        materials.sort()
        return materials

    def hash_materials(self):
        material_hashes = [hash(material) for material in self.materials]
        return short_hash(tuple(material_hashes))

    def get_nearest_neighbours(self, embedding: torch.Tensor):
        distances = torch.cdist(embedding, self.embeddings)
        indices = torch.argmin(distances, dim = -1)
        nearest_neighbours = self.model(torch.stack([self.materials[i].get_coeffs() for batch in indices for i in batch]))
        nearest_neighbours = nearest_neighbours.view(embedding.shape[0], embedding.shape[1], -1)
        return nearest_neighbours

    def embedding_to_refractive_indices(self, embedding: torch.Tensor):
        for batch in embedding:
            for material_embedding in batch:
                assert material_embedding in self.embeddings, f"Material embedding {material_embedding} not found in embeddings. Call get_nearest_neighbours first."
        lookup = self.embeddings.repeat(embedding.shape[0], embedding.shape[1], 1, 1)
        embedding = embedding[:, :, None].repeat(1, 1, self.num_materials, 1)
        # i don't actually know what this is doing
        mask_soft = torch.exp(-torch.abs(embedding - lookup)).prod(dim = -1)
        mask_hard = torch.zeros_like(mask_soft).scatter_(-1, mask_soft.argmax(dim = -1, keepdim = True), 1.0)
        mask = mask_hard + (mask_soft - mask_soft.detach())
        all_refractive_indices = self.materials_refractive_indices.repeat(embedding.shape[0], embedding.shape[1], 1, 1)
        refractive_indices = (all_refractive_indices * mask[:, :, :, None]).sum(dim = -2)
        return refractive_indices

    def encode(self, materials: list[Material]):
        assert len(materials) > 0, "No materials provided"
        assert isinstance(materials[0], Material), "Materials must be of type Material"
        input_features = torch.stack([material.get_coeffs() for material in materials])
        return self.model(input_features)

    def decode(self, embedding: torch.Tensor):
        distances = torch.cdist(embedding, self.embeddings)
        indices = torch.argmin(distances, dim = -1)
        materials = [[self.materials[index] for index in batch] for batch in indices]
        return materials

    def euclidean_distance(self, x, y):
        return torch.norm(x - y, p = 2, dim = -1)

    def compute_loss(self):
        loss = torch.tensor([0.0], device=CM().get('device'))
        num_pairs = 0

        for material_i, material_j in itertools.combinations_with_replacement(self.materials, 2):
            embedding_i = self.model(material_i.get_coeffs()[None, :])
            embedding_j = self.model(material_j.get_coeffs()[None, :])


            embedding_distance = self.euclidean_distance(embedding_i, embedding_j)

            bc_distance = self.euclidean_distance(material_i.get_coeffs()[None, :], material_j.get_coeffs()[None, :]) / self.LOSS_SCALE

            loss += (embedding_distance - bc_distance) ** 2
            num_pairs += 1

        return loss / num_pairs if num_pairs > 0 else torch.tensor(0.0, requires_grad = True, device = CM().get('device'))

    def train(self):
        optimiser = torch.optim.Adam(self.model.parameters(), lr = CM().get('material_embedding.learning_rate'))
        final_loss = None
        for _ in range(CM().get('material_embedding.num_epochs')):
            optimiser.zero_grad()
            loss = self.compute_loss()
            final_loss = loss
            loss.backward()
            optimiser.step()
        if final_loss:
            print(f"Final loss value: {final_loss.item()}")

    def save_embeddings(self):
        torch.save(self.model, self.SAVEPATH)

    def load_embeddings(self):
        try:
            self.model = torch.load(self.SAVEPATH, map_location = CM().get('device'))
        except FileNotFoundError:
            print(f"Saved embeddings not found. Training model.")
            self.train()
            self.save_embeddings()

    def get_materials(self):
        return self.materials

    def get_material(self, title: str):
        try:
            return list(filter(lambda m: m.get_title() == title, self.materials))[0]
        except IndexError:
            raise ValueError(f"Material with title {title} not found")

    def __str__(self):
        max_title_length = max(len(material.get_title()) for material in self.materials)
        material_embeddings = ''
        for material in self.materials:
            embedding = self.model(material.get_coeffs()).cpu().detach().numpy()
            material_embeddings += f"{material.get_title().ljust(max_title_length)}: {embedding}\n"
        return f"Embedding Manager with {len(self.materials)} materials:\n{material_embeddings}"