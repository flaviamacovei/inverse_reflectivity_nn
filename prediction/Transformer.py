import math

import torch
import torch.nn as nn
import sys
sys.path.append(sys.path[0] + '/..')
from prediction.BaseTrainableModel import BaseTrainableModel
from utils.ConfigManager import ConfigManager as CM


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, seq_len: int, dropout: float):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # create matrix of shape (seq_len, embed_dim)
        pe = torch.zeros(seq_len, embed_dim)

        # create a vector of shape (seq_len, 1) representing the position inside the sequence
        position = torch.arange(0, seq_len, dtype = torch.float).unsqueeze(1) # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        # apply sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, seq_len, embed_dim)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)

class LayerNormalisation(nn.Module):
    def __init__(self, eps: float = 10e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # multiplicative
        self.bias = nn.Parameter(torch.zeros(1)) # additive

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h # number of heads
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_k = d_model // h # dimension of a single head
        self.W_q = nn.Linear(d_model, d_model) # query weights
        self.W_k = nn.Linear(d_model, d_model) # key weights
        self.W_v = nn.Linear(d_model, d_model) # value weights

        self.W_o = nn.Linear(d_model, d_model) # output weights
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        # new d_k calculation because this method is static but the value is the same as self.d_k
        d_k = query.shape[-1]

        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # in the tutorial it was replaced by -1e9 not inf but I think inf is cleaner
            attention_scores.masked_fill_(mask == 0, -torch.inf)
        attention_scores = attention_scores.softmax(dim = -1) # (batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores # return output for further calculation but also attention scores for visualisation


    def forward(self, q, k, v, mask):
        query = self.W_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.W_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.W_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, dk) --> (batch, h, seq_len, dk)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k) # contiguous means in-place?

        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.W_o(x)

class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalisation()

    def forward(self, x, sublayer):
        # in original paper: first sublayer then norm but many implementations do first norm then sublayer
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # src_mask is to mask padding tokens but I think in my case there is no input paddings so I don't need it?
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalisation()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # x or src is the input to the decoder
        # encoder_output or tgt is output from encoder
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalisation()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        # (batch, seq_len, in_dim) --> (batch, seq_len, out_dim)
        return self.proj(x)

class BilinearProjectionLayer(nn.Module):
    def __init__(self, d1_in_dim: int, d1_out_dim: int, d2_in_dim: int, d2_out_dim: int):
        super().__init__()
        self.d1_proj = nn.Linear(d1_in_dim, d1_out_dim)
        self.d2_proj = nn.Linear(d2_in_dim, d2_out_dim)

    def forward(self, x):
        # (batch, d1_in_dim, d2_in_dim) --> (batch, d1_in_dim, d2_out_dim)
        x = self.d2_proj(x)
        # (batch, d1_in_dim, d2_out_dim) --> (batch, d2_out_dim, d1_in_dim)
        x = x.transpose(1, 2)
        # (batch, d2_out_dim, d1_in_dim) --> (batch, d2_out_dim, d1_out_dim)
        x = self.d1_proj(x)
        # (batch, d2_out_dim, d1_out_dim) --> (batch, d1_out_dim, d2_out_dim)
        x = x.transpose(1, 2)
        return x

class TrainableTransformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, encoder_projection: BilinearProjectionLayer, encoder_mask_projection: ProjectionLayer, decoder_projection: ProjectionLayer, output_projection: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.encoder_projection = encoder_projection
        self.encoder_mask_projection = encoder_mask_projection
        self.decoder_projection = decoder_projection
        self.output_projection = output_projection

    def disjoin_mask(self, mask):
        mask = self.encoder_mask_projection(mask).to(torch.bool) # (batch, |wl|) --> (batch, hidden_seq_len)
        # disjunction of mask with itself
        repeated_mask = mask[:, :, None].repeat(1, 1, mask.shape[1])
        mask = (repeated_mask | repeated_mask.transpose(-2, -1))[:, None]
        return mask # (batch, 1, hidden_seq_len, hidden_seq_len)

    def encode(self, src, src_mask):
        # src has shape (batch, |wl|, 2)
        src = self.src_pos(src)
        src = self.encoder_projection(src)
        src_mask = self.disjoin_mask(src_mask)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        # tgt has shape (batch, |coating|, material_embed_dim + 1)
        tgt = self.tgt_pos(tgt)
        tgt = self.decoder_projection(tgt)
        src_mask = self.encoder_mask_projection(src_mask)[:, None, None] # (batch, |wl|) --> (batch, 1, 1, hidden_seq_len)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return torch.abs(self.output_projection(x))

class Transformer(BaseTrainableModel):
    def __init__(self):

        # some values:
        # d_model = hyperparameter that needs to be tuned, the dimension in which patterns will be encoded, I think > material_embed_dim + 1
        # d_ff = let's try 2048 but that might be too big

        # src_seq_len = |wl|
        self.src_seq_len = CM().get('wavelengths').shape[0]
        # hyperparameter that needs to be tuned, should be < src_seq_len so that cuda doesn't cry
        self.hidden_seq_len = CM().get('transformer.hidden_seq_len')
        # tgt_seq_len = |coating|
        self.tgt_seq_len = CM().get('num_layers') + 2
        # src_embed_dim = 2 (lower and upper bound)
        self.src_embed_dim = 2
        # tgt_embed_dim = embedding size of coatings
        self.tgt_embed_dim = CM().get('material_embedding.dim') + 1
        # hyperparameter that needs to be tuned
        self.d_model = CM().get('transformer.d_model')
        # num layers (hyperparamter)
        self.N = CM().get('transformer.num_layers')
        # num attn heads
        self.h = CM().get('transformer.num_heads')
        # dropout hyperparameter
        self.dropout = CM().get('transformer.dropout') # maybe move this under training if I can use it in other architectures too
        # feed-forward dimension (hyperparameter)
        self.d_ff = CM().get('transformer.d_ff')


        model = self.build_transformer(self.src_seq_len, self.hidden_seq_len, self.tgt_seq_len, self.src_embed_dim, self.tgt_embed_dim, self.d_model, self.N, self.h, self.dropout, self.d_ff)
        super().__init__(model.to(CM().get('device')))

    def get_model_output(self, src, tgt = None):
        """
        Get output of the model for given input.

        For the Transformer architecture, the necessity of tgt depends on the guidance.

        Args:
            src: Input data.
            tgt: Target data.

        Returns:
            Output of the model.
        """
        src = src.view((src.shape[0], src.shape[1] // 2, 2))  # (batch, 2 * |wl|) --> (batch, |wl|, 2)
        src_mask = torch.ones(src.shape[0], src.shape[1], device = CM().get('device')) # maybe later: mask out masked ranges
        tgt_mask = torch.triu(torch.ones(size = (1, tgt.shape[1], tgt.shape[1]), device = CM().get('device')), diagonal = 1).type(torch.int) == 0
        encoder_output = self.model.encode(src, src_mask)
        decoder_output = self.model.decode(encoder_output, src_mask, tgt, tgt_mask)
        return self.model.project(decoder_output)

    def scale_gradients(self):
        pass


    def build_transformer(self, src_seq_len: int, hidden_seq_len: int, tgt_seq_len: int, src_embed_dim: int = 2, tgt_embed_dim: int = 2, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048):
        # N is the number of blocks per encoder / decoder
        # values from original paper

        # positional encoding layers
        src_pos = PositionalEncoding(src_embed_dim, src_seq_len, dropout)
        tgt_pos = PositionalEncoding(tgt_embed_dim, tgt_seq_len, dropout)

        # source input projection
        src_proj = BilinearProjectionLayer(src_seq_len, hidden_seq_len, src_embed_dim, d_model) # (batch, |wl|, 2) --> (batch, hidden_seq_len, d_model)
        src_mask_proj = ProjectionLayer(src_seq_len, hidden_seq_len) # (batch, |wl|) --> (batch, hiddens_seq_len)

        # encoder blocks
        encoder_blocks = []
        for _ in range(N):
            encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
            feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
            encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
            encoder_blocks.append(encoder_block)

        # target input projection
        tgt_proj = ProjectionLayer(tgt_embed_dim, d_model)

        # decoder blocks
        decoder_blocks = []
        for _ in range(N):
            decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
            decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
            feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
            decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
            decoder_blocks.append(decoder_block)

        # encoder and decoder
        encoder = Encoder(nn.ModuleList(encoder_blocks))
        decoder = Decoder(nn.ModuleList(decoder_blocks))

        # output projection layer
        out_proj = ProjectionLayer(d_model, tgt_embed_dim)

        # transformer
        transformer = TrainableTransformer(encoder, decoder, src_pos, tgt_pos, src_proj, src_mask_proj, tgt_proj, out_proj)

        # initialise params with xavier
        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        return transformer