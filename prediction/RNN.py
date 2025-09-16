import torch
import torch.nn as nn
import sys
sys.path.append(sys.path[0] + '/..')
from utils.ConfigManager import ConfigManager as CM
from data.material_embedding.EmbeddingManager import EmbeddingManager as EM
from prediction.BaseTrainableModel import BaseTrainableModel
from ui.visualise import visualise_matrix

class RNNBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.input_to_hidden = nn.Sequential(
            nn.Linear(self.in_dim + self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
        )
        self.input_to_output = nn.Sequential(
            nn.Linear(self.in_dim + self.hidden_dim, self.out_dim),
            nn.Tanh(),
        )

    def forward(self, x, hidden):
        if hidden is None:
            batch_size = x.shape[0]
            hidden = self.init_hidden().repeat(batch_size, 1)
        combined = torch.cat((x, hidden), dim = 1)
        hidden = self.input_to_hidden(combined)
        output = self.input_to_output(combined)
        return output, hidden

    def init_hidden(self):
        '''Create initial hidden state'''
        return torch.zeros(1, self.hidden_dim).to(CM().get('device'))

class Encoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.rnn = RNNBlock(in_dim, hidden_dim, out_dim)

    def forward(self, x):
        seq_len = x.shape[1]
        hidden = None
        out = None
        for i in range(seq_len):
            out, hidden = self.rnn(x[:, i], hidden)
        return out

class Decoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        
        self.rnn = RNNBlock(in_dim, hidden_dim, out_dim)
    
    def forward(self, x, encoder_output):
        seq_len = x.shape[1]
        hidden = encoder_output
        sequence = None
        for i in range(seq_len):
            out, hidden = self.rnn(x[:, i], hidden)
            sequence = out[:, None] if sequence is None else torch.cat([sequence, out[:, None]], dim = 1)
        return torch.abs(sequence)

class TrainableRNN(nn.Module):
    def __init__(self, enc_in_dim: int, enc_hidden_dim: int, enc_out_dim: int, dec_thickness_in_dim: int, dec_material_in_dim: int, dec_hidden_dim: int, dec_thickness_out_dim: int, dec_material_out_dim: int, flattened_material_dim: int, out_dim: int):
        super().__init__()
        self.encoder = Encoder(enc_in_dim, enc_hidden_dim, enc_out_dim)
        self.decoder_thickness_head = Decoder(dec_thickness_in_dim, dec_hidden_dim, dec_thickness_out_dim)
        self.decoder_material_head = Decoder(dec_material_in_dim, dec_hidden_dim, dec_material_out_dim)
        self.projection = nn.Linear(flattened_material_dim, out_dim)

    def encode(self, src):
        return self.encoder(src)

    def decode(self, encoder_output, tgt):
        thicknesses = tgt[:, :, :1]
        materials = tgt[:, :, 1:]
        decoded_thicknesses = self.decoder_thickness_head(thicknesses, encoder_output)
        decoded_materials = self.decoder_material_head(materials, encoder_output)
        return decoded_thicknesses, decoded_materials

    def project(self, x):
        return self.projection(x)

class RNN(BaseTrainableModel):
    """
    Trainable prediction model using an RNN as base.

    Attributes:
        model: Instance of TrainableRNN.
    """
    def __init__(self):
        """Initialise an MLP instance."""
        super().__init__()

    def build_model(self):
        enc_in_dim = self.src_dim
        enc_hidden_dim = CM().get('rnn.encoder_dim')
        enc_out_dim = CM().get('rnn.decoder_dim')
        dec_thickness_in_dim = self.out_dims['thickness']
        dec_material_in_dim = self.tgt_dim
        dec_hidden_dim = CM().get('rnn.decoder_dim')
        dec_thickness_out_dim = self.out_dims['thickness']
        dec_material_out_dim = self.tgt_dim
        flattened_material_dim = self.out_dims['seq_len'] * self.tgt_dim
        out_dim = self.out_dims['seq_len'] * self.out_dims['material']
        return TrainableRNN(enc_in_dim, enc_hidden_dim, enc_out_dim, dec_thickness_in_dim, dec_material_in_dim, dec_hidden_dim, dec_thickness_out_dim, dec_material_out_dim, flattened_material_dim, out_dim).to(CM().get('device'))

    def get_model_output(self, src, tgt = None):
        """
        Get output of the model for given input.

        Args:
            src: Input data.
            tgt: Target data.

        Returns:
            Output of the model.
        """
        encoder_output = self.model.encode(src) # (batch_size, encoder_hidden_dim)
        if tgt is not None and len(tgt.shape) != 1:
            # in training mode, target is specified
            # in training mode explicit leg, target is dummy data (len(shape) == 1) and should be ignored -> move to inference block
            decoded_thicknesses, decoded_materials = self.model.decode(encoder_output, tgt)
            projected_materials = self.model.project(decoded_materials.flatten(start_dim = 1)).reshape(-1, self.tgt_seq_len, self.tgt_vocab_size)
            return torch.cat([decoded_thicknesses, projected_materials], dim = -1)
        else:
            # in inference mode, target is not specified
            thickness = torch.ones((1, 1, 1)).to(CM().get('device'))
            bos = self.get_bos()
            tgt = torch.cat([thickness, bos], dim = -1).repeat(encoder_output.shape[0], 1, 1)
            while tgt.shape[1] < self.tgt_seq_len:
                decoded_thicknesses, decoded_materials = self.model.decode(encoder_output, tgt)
                next = torch.cat([decoded_thicknesses, decoded_materials], dim = -1)
                next = next[:, -1:] # take only last item but keep dimension
                tgt = torch.cat([tgt, next], dim = 1)
            final_thicknesses = tgt[:, :, :1]
            final_materials = tgt[:, :, 1:]
            projected_materials = self.model.project(final_materials.flatten(start_dim = 1)).reshape(-1, self.tgt_seq_len, self.tgt_vocab_size)
            return torch.cat([final_thicknesses, projected_materials], dim = -1)

    def scale_gradients(self):
        if self.guidance == "free":
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

    def get_architecture_name(self):
        """
        Return name of model architecture.
        """
        return "rnn"