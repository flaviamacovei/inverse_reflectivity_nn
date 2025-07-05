import torch
import torch.nn as nn
import sys
sys.path.append(sys.path[0] + '/..')
from prediction.BaseTrainableModel import BaseTrainableModel
from utils.ConfigManager import ConfigManager as CM

class TrainableTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        input_dim = 2
        target_dim = CM().get('material_embedding.dim') + 1
        d_model = CM().get('transformer.model_dim')
        nhead = CM().get('transformer.num_heads')
        num_layers = CM().get('transformer.num_layers')
        dim_feedforward = CM().get('transformer.d_ff')
        self.max_len_src = CM().get('wavelengths').size()[0]
        self.max_len_tgt = CM().get('layers.max') + 2
        dropout = CM().get('transformer.dropout')

        self.encoder_input_proj = nn.Linear(input_dim, d_model)
        self.decoder_input_proj = nn.Linear(target_dim, d_model)
        self.output_proj = nn.Linear(d_model, target_dim)

        self.pos_encoder = PositionalEncoding(d_model, self.max_len_src)
        self.pos_decoder = PositionalEncoding(d_model, self.max_len_tgt)

        encoder_layer = EncoderLayer(d_model, nhead, dim_feedforward, dropout)
        decoder_layer = DecoderLayer(d_model, nhead, dim_feedforward, dropout)

        self.encoder = Encoder(encoder_layer, num_layers)
        self.decoder = Decoder(decoder_layer, num_layers)

    def forward(self, src, tgt = None, mode = 'free'):
        # src: (B, src_len * input_dim)
        # tgt: (B, tgt_len, target_dim) or None

        # reshape to (B, src_len, input_dim)
        src = src.reshape(src.shape[0], CM().get('wavelengths').size()[0], -1)

        src = self.encoder_input_proj(src)  # (B, src_len, d_model)
        src = self.pos_encoder(src)

        memory = self.encoder(src)

        if mode == 'guided' and tgt is not None:
            tgt = self.decoder_input_proj(tgt)
        else:
            # Free mode: use zeros or learned start token
            batch_size, seq_len = src.size(0), src.size(1)
            tgt = torch.zeros(batch_size, self.max_len_tgt, self.output_proj.out_features, device=src.device)
            tgt = self.decoder_input_proj(tgt)

        tgt = self.pos_decoder(tgt)
        output = self.decoder(tgt, memory)
        return self.output_proj(output)  # (B, tgt_len, target_dim)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        # x: (B, T, D)
        seq_len = x.size(1)
        return x + self.pos_embedding[:, :seq_len, :]

class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.norm = nn.LayerNorm(encoder_layer.d_model)

    def forward(self, src):
        output = src
        for layer in self.layers:
            output = layer(output)
        return self.norm(output)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)

class Decoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.norm = nn.LayerNorm(decoder_layer.d_model)

    def forward(self, tgt, memory):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory)
        return self.norm(output)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory):
        tgt2, _ = self.self_attn(tgt, tgt, tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2, _ = self.cross_attn(tgt, memory, memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(torch.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return self.norm3(tgt)

class Transformer(BaseTrainableModel):
    def __init__(self):
        model = TrainableTransformer()
        print(f"number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        super().__init__(model.to(CM().get('device')))

    def get_model_output(self, src, tgt = None, guidance = 'free'):
        """
        Get output of the model for given input.

        For the Transformer architecture, the necessity of tgt depends on the guidance.

        Args:
            src: Input data.
            tgt: Target data.
            guidance: Guidance data.

        Returns:
            Output of the model.
        """
        return self.model(src, tgt, guidance)

    def scale_gradients(self):
        pass
