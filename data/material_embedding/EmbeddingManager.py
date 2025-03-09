import torch
import torch.nn as nn
import itertools
import yaml
import sys
sys.path.append(sys.path[0] + '/..')
from data.values.Material import Material
from utils.ConfigManager import ConfigManager as CM

class EmbeddingManager(nn.Module):
    def __init__(self):
        super().__init__()
        self.materials = self.load_materials()
        self.num_materials = len(self.materials)
        self.embedding_dim = CM().get('material_embedding.dim')

        self.embeddings = nn.Parameter(torch.randn(self.num_materials, self.embedding_dim))

        input_dim = self.materials[0].get_B().shape[0] * 2

        self.net = nn.Linear(input_dim, self.embedding_dim)

        self.SAVEPATH = 'data/material_embedding/embeddings.pt'

    def load_materials(self, ):
        with open(CM().get('material_embedding.data_file'), 'r') as f:
            data = yaml.safe_load(f)
        return [Material(m['title'], m['B'], m['C']) for m in data['materials']]

    def forward(self, material_index):
        return self.embeddings[material_index]

    def map_to_embedding(self, B, C):
        input_features = torch.cat((B, C), dim = -1)
        return self.net(input_features)

    def euclidean_distance(self, x, y):
        return torch.norm(x - y, p = 2, dim = -1)

    def compute_loss(self):
        loss = torch.tensor(0.0, device=CM().get('device'))
        num_pairs = 0

        for (i, material_i), (j, material_j) in itertools.combinations(enumerate(self.materials), 2):
            embedding_i = self.forward(i)
            embedding_j = self.forward(j)

            embedding_distance = self.euclidean_distance(embedding_i, embedding_j)

            bc_distance = self.euclidean_distance(torch.cat([material_i.get_B(), material_i.get_C()]), torch.cat([material_j.get_B(), material_j.get_C()]))

            loss += (embedding_distance - bc_distance) ** 2
            num_pairs += 1

        return loss / num_pairs if num_pairs > 0 else torch.tensor(0.0, requires_grad = True, device = CM().get('device'))

    def train(self):
        optimiser = torch.optim.Adam(self.parameters(), lr = CM().get('material_embedding.learning_rate'))
        for epoch in range(CM().get('material_embedding.num_epochs')):
            optimiser.zero_grad()
            loss = self.compute_loss()
            loss.backward()
            optimiser.step()
            if epoch % 100 == 0:
                print(f'Epoch {epoch}: loss = {loss}')

    def save_embeddings(self):
        torch.save(self.embeddings, self.SAVEPATH)

    def load_embeddings(self):
        self.embeddings = torch.load(self.SAVEPATH)