import math
import torch
import torch.nn as nn
import sys
sys.path.append(sys.path[0] + '/..')
from prediction.BaseTrainableModel import ThicknessPostProcess
from prediction.BaseSequentialModel import BaseSequentialModel
from utils.ConfigManager import ConfigManager as CM
from utils.math_utils import largest_prime_factor


class DownSize(nn.Module):
    def __init__(self, start_length: int, target_length: int, downsize_dim: int = 1):
        super().__init__()
        self.start_length = start_length
        self.target_length = target_length
        self.downsize_dim = downsize_dim

class DownSizeSample(DownSize):
    """
    Sample items from sequence at equal distance
    """
    def __init__(self, start_length: int, target_length: int, downsize_dim: int = 1):
        super().__init__(start_length, target_length, downsize_dim)

    def forward(self, x):
        difference = self.start_length % self.target_length
        max_sample = self.start_length - difference
        step_size = math.ceil(max_sample / self.target_length)
        indices = torch.arange(0, max_sample, step_size, dtype = torch.long, device = CM().get('device')) + difference // 2
        return x.index_select(index = indices, dim = self.downsize_dim)

class DownSizeWindow(DownSize):
    """
    Learned weights for windows over sequence
    """
    def __init__(self, start_length: int, target_length: int, downsize_dim: int = 1):
        super().__init__(start_length, target_length, downsize_dim)
        self.num_windows = largest_prime_factor(self.start_length)
        self.encoding = nn.Sequential(
            nn.Linear(self.start_length // self.num_windows, self.target_length),
            nn.ReLU()
        )

    def forward(self, x):
        windows = x.chunk(self.num_windows, self.downsize_dim)
        encoded_windows = None
        for window in windows:
            encoded_window = self.encoding(window.transpose(self.downsize_dim, -1)).transpose(self.downsize_dim, -1)
            encoded_windows = encoded_window[:, None] if encoded_windows is None else torch.cat([encoded_windows, encoded_window[:, None]], dim = 1)
        return encoded_windows.mean(dim = 1)

class DownSizeMean(DownSize):
    """
    Mean pooling
    """
    def __init__(self, start_length: int, target_length: int, downsize_dim: int = 1):
        super().__init__(start_length, target_length, downsize_dim)
        kernel_size = self.start_length - self.target_length + 1
        stride = 1
        padding = 0
        self.mean_pool = nn.AvgPool1d(kernel_size = kernel_size, stride = stride, padding = padding)

    def forward(self, x):
        return self.mean_pool(x.transpose(self.downsize_dim, -1)).transpose(self.downsize_dim, -1)

class DownSizeMax(DownSize):
    """
    Max pooling
    """

    def __init__(self, start_length: int, target_length: int, downsize_dim: int = 1):
        super().__init__(start_length, target_length, downsize_dim)
        kernel_size = self.start_length - self.target_length + 1
        stride = 1
        padding = 0
        self.mean_pool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.mean_pool(x.transpose(self.downsize_dim, -1)).transpose(self.downsize_dim, -1)

class DownSizeConv(DownSize):
    """
    Learned 1D convolution
    """

    def __init__(self, start_length: int, target_length: int, downsize_dim: int = 1):
        super().__init__(start_length, target_length, downsize_dim)
        kernel_size = self.start_length - self.target_length + 1
        stride = 1
        padding = 0
        self.mean_pool = nn.Conv1d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.mean_pool(x.transpose(self.downsize_dim, -1)).transpose(self.downsize_dim, -1)

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, seq_len: int, dropout: float):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # create matrix of shape (seq_len, embed_dim)
        pe = torch.zeros(seq_len, embed_dim, device = CM().get('device'))

        # create a vector of shape (seq_len, 1) representing the position inside the sequence
        position = torch.arange(0, seq_len, dtype = torch.float, device = CM().get('device')).unsqueeze(1) # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, embed_dim).float().to(CM().get('device')) * (-math.log(10000.0) / embed_dim))

        # apply sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term[0::2])
        pe[:, 1::2] = torch.cos(position * div_term[1::2])

        pe = pe.unsqueeze(0) # (1, seq_len, embed_dim)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :])
        return self.dropout(x)

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Embedding(seq_len, embed_dim)

    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)  # (1, seq_len)
        x = x + self.pe(positions)  # broadcast add
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
            attention_scores.masked_fill_(mask == 0, -torch.inf)
        attention_scores = attention_scores.softmax(dim = -1) # (batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores # return output for further calculation but also attention scores for visualisation

    def forward(self, q, k, v, mask = None):
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

    def forward(self, x):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalisation()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalisation()

    def forward(self, x, encoder_output, mask):
        for layer in self.layers:
            x = layer(x, encoder_output, mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        # (batch, seq_len, in_dim) --> (batch, seq_len, out_dim)
        return self.proj(x)

class TrainableTransformer(nn.Module):
    def __init__(self, encoder: Encoder, thicknesses_decoder: Decoder, materials_decoder: Decoder, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, encoder_downsize: DownSize, encoder_projection: ProjectionLayer, decoder_projection: ProjectionLayer, thickness_out: nn.Module, material_out: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.thicknesses_decoder = thicknesses_decoder
        self.materials_decoder = materials_decoder
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.encoder_downsize = encoder_downsize
        self.encoder_projection = encoder_projection
        self.decoder_projection = decoder_projection
        self.thickness_out = thickness_out
        self.material_out = material_out


class Transformer(BaseSequentialModel):
    def __init__(self):
        super().__init__()

    def build_model(self):
        # some values:
        # d_model = hyperparameter that needs to be tuned, the dimension in which patterns will be encoded, I think > material_embed_dim + 1
        # d_ff = let's try 2048 but that might be too big

        # hyperparameter that needs to be tuned, should be < src_seq_len so that cuda doesn't cry
        self.hidden_seq_len = CM().get('transformer.hidden_seq_len')
        # hyperparameter that needs to be tuned
        self.d_model = CM().get('transformer.d_model')
        # num layers (hyperparamter)
        self.N = CM().get('transformer.num_layers')
        # num attn heads
        self.h = CM().get('transformer.num_heads')
        # dropout hyperparameter
        self.dropout = CM().get(
            'transformer.dropout')  # maybe move this under training if I can use it in other architectures too
        # feed-forward dimension (hyperparameter)
        self.d_ff = CM().get('transformer.d_ff')

        downsize_classes = {
            'sample': DownSizeSample,
            'window': DownSizeWindow,
            'mean': DownSizeMean,
            'max': DownSizeMax,
            'conv': DownSizeConv
        }

        # positional encoding layers
        src_pos = PositionalEncoding(self.src_dim, self.src_seq_len, self.dropout)
        tgt_pos = PositionalEncoding(self.tgt_dim, self.tgt_seq_len - 1, self.dropout)

        # source input projection
        SrcDownsizeClass = downsize_classes[CM().get('transformer.downsize')]
        src_downsize = SrcDownsizeClass(self.src_seq_len, self.hidden_seq_len, 1)
        src_proj = ProjectionLayer(self.src_dim, self.d_model) # (batch, hidden_seq_len, 2) --> (batch, hidden_seq_len, d_model)

        # encoder blocks
        encoder_blocks = []
        for _ in range(self.N):
            encoder_self_attention_block = MultiHeadAttentionBlock(self.d_model, self.h, self.dropout)
            feed_forward_block = FeedForwardBlock(self.d_model, self.d_ff, self.dropout)
            encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, self.dropout)
            encoder_blocks.append(encoder_block)

        # target input projection
        tgt_proj = ProjectionLayer(self.tgt_dim, self.d_model)

        # decoder blocks
        thicknesses_decoder_blocks = []
        for _ in range(self.N):
            decoder_self_attention_block = MultiHeadAttentionBlock(self.d_model, self.h, self.dropout)
            decoder_cross_attention_block = MultiHeadAttentionBlock(self.d_model, self.h, self.dropout)
            feed_forward_block = FeedForwardBlock(self.d_model, self.d_ff, self.dropout)
            decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, self.dropout)
            thicknesses_decoder_blocks.append(decoder_block)

        materials_decoder_blocks = []
        for _ in range(self.N):
            decoder_self_attention_block = MultiHeadAttentionBlock(self.d_model, self.h, self.dropout)
            decoder_cross_attention_block = MultiHeadAttentionBlock(self.d_model, self.h, self.dropout)
            feed_forward_block = FeedForwardBlock(self.d_model, self.d_ff, self.dropout)
            decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, self.dropout)
            materials_decoder_blocks.append(decoder_block)

        # encoder and decoder
        encoder = Encoder(nn.ModuleList(encoder_blocks))
        thicknesses_decoder = Decoder(nn.ModuleList(thicknesses_decoder_blocks))
        materials_decoder = Decoder(nn.ModuleList(materials_decoder_blocks))

        # output projection layer
        thickness_out = nn.Sequential(
            ProjectionLayer(self.d_model, self.out_dims['thickness']),
            ThicknessPostProcess(self.out_dims['seq_len'] - 1)
        )
        material_out = ProjectionLayer(self.d_model, self.out_dims['material'])

        # transformer
        transformer = TrainableTransformer(encoder, thicknesses_decoder, materials_decoder, src_pos, tgt_pos, src_downsize, src_proj, tgt_proj, thickness_out, material_out)

        # initialise params with xavier
        for p in transformer.parameters():
            if p.dim() > 1:
                # nn.init.xavier_uniform_(p)
                nn.init.kaiming_uniform_(p)

        return transformer.to(CM().get('device'))

    def project_encoder(self, src):
        src = self.model.src_pos(src) # (batch, |wl|, 2)
        src = self.model.encoder_downsize(src) # (batch, hidden_seq_len, 2)
        src = self.model.encoder_projection(src) # (batch, hidden_seq_len, d_model)
        return src

    def project_decoder(self, tgt):
        # tgt has shape (batch, |coating|, material_embed_dim + 1)
        tgt = self.model.tgt_pos(tgt)
        tgt = self.model.decoder_projection(tgt)
        return tgt

    def encode(self, src):
        output = self.model.encoder(src)
        # return torch.randn_like(output)
        return output

    def decode(self, encoder_output, tgt, mask = None):
        thicknesses = self.model.thicknesses_decoder(tgt, encoder_output, mask)
        materials = self.model.materials_decoder(tgt, encoder_output, mask)
        return thicknesses, materials

    def output_thicknesses(self, decoder_output):
        return self.model.thickness_out(decoder_output)

    def output_materials(self, decoder_output):
        return self.model.material_out(decoder_output)

    def make_tgt_mask(self, tgt):
        if CM().get('transformer.tgt_struct_mask'):
            substrate = self.get_bos()
            air = self.get_eos()
            masked_materials_encoding = torch.cat([substrate, air], dim = -1) # (1, 1, 2)
            struct_mask = tgt[:, :, 1][:, :, None].repeat(1, 1, 2) == masked_materials_encoding
            struct_mask = torch.logical_or(struct_mask[:, :, 0], struct_mask[:, :, 1])[:, :, None] # (batch, |coating|, 1)
            struct_mask_repeated = struct_mask.repeat(1, 1, tgt.shape[1])
            tgt_struct_mask = ~struct_mask_repeated & ~struct_mask_repeated.transpose(-2, -1) # (batch, |coating|, |coating|)
            tgt_struct_mask.diagonal(dim1 = 1, dim2 = 2).copy_(1) # each position is unmasked wrt itself
        else:
            tgt_struct_mask = torch.ones((tgt.shape[0], tgt.shape[1], tgt.shape[1]), device = CM().get('device')) # (batch, |coating|, |coating|)
        if CM().get('transformer.tgt_caus_mask'):
            tgt_caus_mask = self.causal_mask(tgt.shape[1]) # (1, |coating|, |coating|)
        else:
            tgt_caus_mask = torch.ones((tgt.shape[0], tgt.shape[1], tgt.shape[1]), device = CM().get('device')) # (batch, |coating|, |coating|)
        return (tgt_struct_mask.to(torch.bool) & tgt_caus_mask.to(torch.bool))[:, None]

    def causal_mask(self, size):
        return torch.triu(torch.ones(size=(1, size, size), device=CM().get('device')), diagonal=1).type(torch.int) == 0

    def get_architecture_name(self):
        """
        Return name of model architecture.
        """
        return "transformer"

    def get_shared_params(self):
        params = []
        for param in self.model.src_pos.parameters():
            params.append(param)
        for param in self.model.encoder_projection.parameters():
            params.append(param)
        for param in self.model.encoder_downsize.parameters():
            params.append(param)
        for param in self.model.encoder.parameters():
            params.append(param)
        for param in self.model.tgt_pos.parameters():
            params.append(param)
        for param in self.model.decoder_projection.parameters():
            params.append(param)
        return params

    def get_thicknesses_params(self):
        params = []
        for param in self.model.thicknesses_decoder.parameters():
            params.append(param)
        for param in self.model.thickness_out.parameters():
            params.append(param)
        return params

    def get_materials_params(self):
        params = []
        for param in self.model.materials_decoder.parameters():
            params.append(param)
        for param in self.model.material_out.parameters():
            params.append(param)
        return params