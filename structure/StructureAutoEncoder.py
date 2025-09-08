import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append(sys.path[0] + '/..')
from utils.ConfigManager import ConfigManager as CM
from data.dataloaders.DynamicDataloader import DynamicDataloader
from data.material_embedding.EmbeddingManager import EmbeddingManager as EM
from data.values.Coating import Coating
from ui.cl_interact import ding


class TrainableAutoEncoder(nn.Module):
    def __init__(self, seq_len: int, embed_dim: int, latent_dim: int, vocab_size: int, num_layers: int):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.in_dim = self.seq_len * self.embed_dim
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.out_dim = self.seq_len * self.vocab_size

        # TODO: logarithmic scale
        self.encoder_dims = [self.latent_dim + (self.in_dim - self.latent_dim) // self.num_layers * i for i in range(self.num_layers + 1)]
        self.encoder_dims.reverse()
        self.encoder_dims[0] = self.in_dim

        encoder_layers = []
        for i in range(len(self.encoder_dims) - 1):
            encoder_layers.append(
                nn.Sequential(
                    nn.Linear(self.encoder_dims[i], self.encoder_dims[i + 1]),
                    nn.ReLU()
                )
            )
        self.encoder = nn.ModuleList(encoder_layers)

        self.decoder_dims = [self.out_dim + (self.latent_dim - self.out_dim) // self.num_layers * i for i in range(self.num_layers + 1)]
        self.decoder_dims.reverse()
        self.decoder_dims[0] = self.latent_dim

        decoder_layers = []
        for i in range(len(self.decoder_dims) - 1):
            decoder_layers.append(
                nn.Sequential(
                    nn.Linear(self.decoder_dims[i], self.decoder_dims[i + 1]),
                    nn.ReLU()
                )
            )
        self.decoder = nn.ModuleList(decoder_layers)

    def encode(self, x):
        for encoder_layer in self.encoder:
            x = encoder_layer(x)
        return x

    def decode(self, x):
        for decoder_layer in self.decoder:
            x = decoder_layer(x)
        return x


class StructureAutoEncoder():
    def __init__(self):
        self.seq_len = CM().get('num_layers') + 2
        self.embed_dim = CM().get('material_embedding.dim')
        latent_dim = CM().get('autoencoder.latent_dim')
        self.vocab_size = len(CM().get('materials.thin_films')) + 2
        num_layers = CM().get('autoencoder.num_layers')

        self.batch_size = 256
        self.learning_rate = 1e-3
        self.num_epochs = 100

        self.autoencoder = TrainableAutoEncoder(self.seq_len, self.embed_dim, latent_dim, self.vocab_size, num_layers)
        self.autoencoder = self.autoencoder.to(CM().get('device'))

    def train(self):
        dataloader = DynamicDataloader(self.batch_size, True)
        dataloader.load_leg(0)

        optimiser = torch.optim.Adam(self.autoencoder.parameters(), lr = self.learning_rate)

        for epoch in range(self.num_epochs):
            epoch_loss = torch.tensor([0.], device =CM().get('device'))
            for batch in dataloader:
                self.autoencoder.train()
                optimiser.zero_grad()
                coating = batch[1][:, :, 1:]
                input = coating.flatten(start_dim = 1).to(CM().get('device'))

                latent_vector = self.autoencoder.encode(input)

                logits = self.autoencoder.decode(latent_vector)
                preds = self.decode(logits)

                loss = self.compute_loss(preds, coating)
                epoch_loss += loss
                loss.backward()
                optimiser.step()
            if epoch % (self.num_epochs / 20) == 0:
                gt = dataloader[0][1][None]
                thicknesses = gt[:, :, 0]
                materials = gt[:, :, 1]
                print(gt.shape)
                latent_vector = self.autoencoder.encode(materials)
                logits = self.autoencoder.decode(latent_vector)
                preds = self.decode(logits)
                print(f"loss in epoch {epoch}: {epoch_loss.item()}\n{Coating(torch.cat([thicknesses[:, :, None], preds], dim = -1))}")

    def compute_loss(self, preds: torch.Tensor, label: torch.Tensor):
        # mse loss
        return torch.sum((preds - label)**2)

    def decode(self, logits: torch.Tensor):
        logits = logits.reshape(logits.shape[0], self.seq_len, self.vocab_size)
        softmax_probabilities = F.softmax(logits, dim=-1)
        return softmax_probabilities @ EM().get_embeddings()


if __name__ == '__main__':
    model = StructureAutoEncoder()
    model.train()
    ding()