import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
import sys

sys.path.append(sys.path[0] + '/..')
from prediction.BaseTrainableModel import BaseTrainableModel
from data.dataloaders.BaseDataloader import BaseDataloader
from utils.ConfigManager import ConfigManager as CM


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()
        assert model_dim % num_heads == 0, "Model dimension must be evenly divisible by number of attention heads"

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.d_k = model_dim // num_heads

        self.W_q = nn.Linear(model_dim, model_dim)
        self.W_k = nn.Linear(model_dim, model_dim)
        self.W_v = nn.Linear(model_dim, model_dim)
        self.W_o = nn.Linear(model_dim, model_dim)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        attention_probabilities = torch.softmax(attention_scores, dim=1)

        output = torch.matmul(attention_probabilities, V)
        return output

    def split_heads(self, x):
        batch_size, sequence_length, model_dim = x.size()
        return x.view(batch_size, sequence_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, sequence_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.model_dim)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)

        output = self.W_o(self.combine_heads(attention_output))
        return output

class PositionWiseFeedForward(nn.Module):
    def __init__(self, model_dim, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(model_dim, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, model_dim)
        )

    def forward(self, x):
        return self.net(x)

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_sequence_length):
        super().__init__()

        positional_encoding = torch.zeros(max_sequence_length, model_dim)
        position = torch.arange(0, max_sequence_length).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * -(math.log(10000.0) / model_dim))

        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('positional_encoding', positional_encoding.unsqueeze(0))

    def forward(self, x):
        return x + self.positional_encoding[:, :x.size(1)]

class EncoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attention = MultiHeadAttention(model_dim, num_heads)
        self.feed_forward = PositionWiseFeedForward(model_dim, d_ff)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attention_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, model_dim, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attention = MultiHeadAttention(model_dim, num_heads)
        self.cross_attention = MultiHeadAttention(model_dim, num_heads)
        self.feed_forward = PositionWiseFeedForward(model_dim, d_ff)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoding_output, source_mask, target_mask):
        attention_output = self.self_attention(x, x, x, target_mask)
        x = self.norm1(x + self.dropout(attention_output))
        attention_output = self.cross_attention(x, encoding_output, encoding_output, source_mask)
        x = self.norm2(x + self.dropout(attention_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class TrainableTransformer(nn.Module):

    # TODO: rename vocab size
    def __init__(self, source_vocab_size, target_vocab_size, model_dim, num_heads, num_layers, d_ff, max_sequence_length, dropout):
        super().__init__()
        self.output_size = (CM().get('layers.max') + 2) * (CM().get('material_embedding.dim') + 1)
        self.encoder_embedding = nn.Embedding(source_vocab_size, model_dim)
        self.decoder_embedding = nn.Embedding(target_vocab_size, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, max_sequence_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(model_dim, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(model_dim, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(model_dim, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, source, target):
        source_mask = (source != 0).unsqueeze(1).unsqueeze(2)
        target_mask = (target != 0).unsqueeze(1).unsqueeze(3)
        sequence_length = target.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, sequence_length, sequence_length, device = CM().get('device')), diagonal = 1)).bool()
        target_mask = target_mask & nopeak_mask
        return source_mask, target_mask

    def scale_input(self, x):
        # TODO: find a more accurate scaling
        # print(x)
        scaled_x = x * 1e2
        scaled_x = scaled_x.to(torch.int64)
        # print(scaled_x)
        # print(f"is this equal: {torch.equal(x, (scaled_x / 1e9))}")
        return scaled_x

    def forward(self, x):
        x = self.scale_input(x)
        source = x
        target = x
        source_mask, target_mask = self.generate_mask(source, target)
        source_embedded = self.encoder_embedding(source)
        source_positioned = self.positional_encoding(source_embedded)
        source_embedded = self.dropout(source_positioned)
        target_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(target)))

        encoder_output = source_embedded
        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(encoder_output, source_mask)

        decoder_output = target_embedded
        for decoder_layer in self.decoder_layers:
            decoder_output = decoder_layer(decoder_output, encoder_output, source_mask, target_mask)

        decoder_output = decoder_output.sum(dim = 1)
        output = self.fc(decoder_output)
        return output



    def get_output_size(self):
        return self.output_size


class Transformer(BaseTrainableModel):
    def __init__(self):
        torch.autograd.set_detect_anomaly(True)
        source_vocab_size = int(1e3)
        target_vocab_size = (CM().get('layers.max') + 2) * (CM().get('material_embedding.dim') + 1)
        # TODO: fix this when scale_input is fixed (the factor here is related to one by which the inputs are scaled)
        model_dim = CM().get('transformer.model_dim')
        num_heads = CM().get('transformer.num_heads')
        num_layers = CM().get('transformer.num_layers')
        d_ff = CM().get('transformer.d_ff')
        max_sequence_length = CM().get('wavelengths').shape[0] * 2
        dropout = CM().get('transformer.dropout')
        super().__init__(TrainableTransformer(
            source_vocab_size,
            target_vocab_size,
            model_dim,
            num_heads,
            num_layers,
            d_ff,
            max_sequence_length,
            dropout
        ).to(CM().get('device')))

    def scale_gradients(self):
        pass
