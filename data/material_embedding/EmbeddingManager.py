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
        if self.fixed_points and x in self.fixed_points:
            return self.fixed_points[x]
        return self.net(x)

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

        indim = self.materials[0].get_coeffs().shape[0]

        substrate = self.get_material(CM().get('materials.substrate')).get_coeffs()
        air = self.get_material(CM().get('materials.air')).get_coeffs()
        fixed_points = {
            substrate: torch.zeros((CM().get('material_embedding.dim')), device = CM().get('device')),
            air: torch.ones((CM().get('material_embedding.dim')), device = CM().get('device')) * 10
        }
        self.model = EmbeddingModel(indim, self.num_materials, CM().get('material_embedding.dim'), fixed_points).to(CM().get('device'))

        self.SAVEPATH = f'data/material_embedding/embeddings_{self.hash_materials()}.pt'
        self.load_embeddings()

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
        return hash(tuple(material_hashes))

    def encode(self, materials: list[Material]):
        assert len(materials) > 0, "No materials provided"
        assert isinstance(materials[0], Material), "Materials must be of type Material"
        input_features = torch.stack([material.get_coeffs() for material in materials])
        return self.model(input_features)

    def decode(self, embeddings: torch.Tensor):
        distances = torch.tensor([[self.euclidean_distance(self.model(self.materials[i].get_coeffs()), embeddings[j]) for i in range(self.num_materials)] for j in range(len(embeddings))])
        indices = torch.argmin(distances, dim = -1)
        materials = [self.materials[index] for index in indices]
        return materials, [distances[i, indices[i]].item() for i in range(len(embeddings))]

    def euclidean_distance(self, x, y):
        return torch.norm(x - y, p = 2, dim = -1)

    def compute_loss(self):
        loss = torch.tensor(0.0, device=CM().get('device'))
        num_pairs = 0

        for material_i, material_j in itertools.combinations(self.materials, 2):
            embedding_i = self.model(material_i.get_coeffs())
            embedding_j = self.model(material_j.get_coeffs())

            embedding_distance = self.euclidean_distance(embedding_i, embedding_j)

            bc_distance = self.euclidean_distance(material_i.get_coeffs(), material_j.get_coeffs())

            loss += (embedding_distance - bc_distance) ** 2
            num_pairs += 1

        return loss / num_pairs if num_pairs > 0 else torch.tensor(0.0, requires_grad = True, device = CM().get('device'))

    def train(self):
        optimiser = torch.optim.Adam(self.model.parameters(), lr = CM().get('material_embedding.learning_rate'))
        for epoch in range(CM().get('material_embedding.num_epochs')):
            optimiser.zero_grad()
            loss = self.compute_loss()
            loss.backward()
            optimiser.step()

    def save_embeddings(self):
        torch.save(self.model, self.SAVEPATH)

    def load_embeddings(self):
        try:
            self.model = torch.load(self.SAVEPATH)
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