import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from triton.language import zeros_like

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
        self.in_dim = self.embed_dim
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.out_dim = self.vocab_size

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
        # latent_dim = self.seq_len * CM().get('autoencoder.latent_dim')
        latent_dim = CM().get('autoencoder.latent_dim')
        self.vocab_size = len(CM().get('materials.thin_films')) + 2
        num_layers = CM().get('autoencoder.num_layers')

        self.batch_size = 256
        self.learning_rate = 1e-3
        self.num_epochs = 100

        self.autoencoder = TrainableAutoEncoder(self.seq_len, self.embed_dim, latent_dim, self.vocab_size, num_layers)
        self.autoencoder = self.autoencoder.to(CM().get('device'))

        own_path = os.path.realpath(__file__)
        self.model_path = os.path.join(os.path.dirname(os.path.dirname(own_path)), "out/autoencoder_unmasked.pt")
        self.load()


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
                # input = coating.flatten(start_dim = 1).to(CM().get('device'))
                input = coating.to(CM().get('device'))

                latent_vector = self.autoencoder.encode(input)

                logits = self.autoencoder.decode(latent_vector)
                # logits = self.mask_logits(logits)
                softmax_probabilities = F.softmax(logits, dim = -1)

                # # mse loss
                # preds = self.decode(logits)
                # loss = self.compute_loss(preds, coating)

                # ce loss
                target = self.get_coating_indices(coating)
                loss = F.cross_entropy(softmax_probabilities.view(-1, self.vocab_size), target.view(-1))

                epoch_loss += loss
                loss.backward()
                optimiser.step()
            if epoch % (self.num_epochs / 20) == 0:
                gt = dataloader[0][1][None]
                thicknesses = gt[:, :, :1]
                materials = gt[:, :, 1:]
                latent_vector = self.autoencoder.encode(materials)
                logits = self.autoencoder.decode(latent_vector)
                # logits = self.mask_logits(logits)
                preds = self.logits_to_materials(logits)
                print(f"loss in epoch {epoch}: {epoch_loss.item()}\n{Coating(torch.cat([thicknesses, preds], dim = -1))}")

    def get_coating_indices(self, coating: torch.Tensor):
        coating = coating[:, :, None].repeat(1, 1, self.vocab_size, 1) # (batch, seq_len, vocab_size, embed_dim)
        coating_eq = coating.eq(EM().get_embeddings()) # (batch, seq_len, vocab_size, embed_dim)
        coating_indices = coating_eq.prod(dim = -1) # (batch, seq_len, vocab_size)
        return coating_indices.argmax(dim = -1) # (batch, seq_len)


    def mask_logits(self, logits: torch.Tensor):
        logits = logits.reshape(logits.shape[0], self.seq_len, self.vocab_size)
        substrate = EM().get_material(CM().get('materials.substrate'))
        air = EM().get_material(CM().get('materials.air'))
        substrate_encoding, air_encoding = EM().encode([substrate, air])
        # get index of substrate and air in embeddings lookup
        substrate_index = EM().get_embeddings().eq(substrate_encoding).nonzero(as_tuple = True)[0].item()
        air_index = EM().get_embeddings().eq(air_encoding).nonzero(as_tuple = True)[0].item()

        # create substrate mask
        substrate_position_mask = torch.zeros_like(logits)
        # mask all other tokens at first position
        substrate_position_mask[:, 0] = 1
        # mask substrate at all other positions
        substrate_index_mask = torch.zeros_like(logits)
        substrate_index_mask[:, :, substrate_index] = 1
        substrate_mask = torch.logical_xor(substrate_position_mask, substrate_index_mask)

        # get index of last material to bound air block
        not_air = logits.argmax(dim = -1, keepdim = True).ne(air_index) # (batch, |coating|, |materials_embedding|)
        # logical or along dimension -1
        not_air = not_air.int().sum(dim=-1).bool()  # (batch, |coating|)
        not_air_rev = not_air.flip(dims=[1]).to(torch.int)  # (batch, |coating|)
        last_mat_idx_rev = torch.argmax(not_air_rev, dim=-1)  # (batch)
        last_mat_idx = not_air.shape[-1] - last_mat_idx_rev - 1  # (batch)
        # ensure at least last element is air
        last_mat_idx = torch.minimum(last_mat_idx, torch.tensor(not_air.shape[-1] - 2))
        # create air mask
        range_coating = torch.arange(self.seq_len, device=CM().get('device'))  # (|coating|)
        # mask all other tokens at air block
        air_position_mask = range_coating[None] >= (last_mat_idx + 1)[:, None]  # (batch, |coating|)
        air_position_mask = air_position_mask[:, :, None].repeat(1, 1, self.vocab_size) # (batch, |coating|, |vocab|)
        # mask air at all other positions
        air_index_mask = torch.zeros_like(logits)
        air_index_mask[:, :, air_index] = 1
        air_mask = torch.logical_xor(air_position_mask, air_index_mask)

        # mask logits
        mask = torch.logical_or(substrate_mask, air_mask)
        subtrahend = torch.zeros_like(logits)
        subtrahend[mask] = torch.inf
        logits = logits - subtrahend
        # logits.masked_fill_(mask == 1, -torch.inf)
        return logits
        print("<3")

    def compute_loss(self, preds: torch.Tensor, label: torch.Tensor):
        # mse loss
        return F.cross_entropy(input = preds, target = label)

    def logits_to_materials(self, logits: torch.Tensor):
        logits = logits.reshape(logits.shape[0], self.seq_len, self.vocab_size)
        softmax_probabilities = F.softmax(logits, dim=-1)
        return softmax_probabilities @ EM().get_embeddings()

    def decode(self, latent_vector: torch.Tensor):
        logits = self.autoencoder.decode(latent_vector)
        masked_logits = self.mask_logits(logits)
        return self.logits_to_materials(masked_logits)

    def encode(self, materials: torch.Tensor):
        # latent_materials = self.autoencoder.encode(materials.flatten(start_dim = 1))
        return self.autoencoder.encode(materials)

    def load(self):
        if os.path.exists(self.model_path):
            tmp = torch.load(self.model_path, weights_only = False)
            self.autoencoder = tmp
            self.autoencoder = self.autoencoder.to(CM().get('device'))
        else:
            print("No autoencoder found. Training model...")
            self.train()
            torch.save(self.autoencoder, self.model_path)

if __name__ == '__main__':
    model = StructureAutoEncoder()
    model.train()
    ding()