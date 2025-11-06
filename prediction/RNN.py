import torch
import torch.nn as nn
import sys
sys.path.append(sys.path[0] + '/..')
from utils.ConfigManager import ConfigManager as CM
from prediction.BaseTrainableModel import ThicknessPostProcess
from prediction.BaseSequentialModel import BaseSequentialModel
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
    def __init__(self, d_model: int, hidden_dims: list[int]):
        super().__init__()
        blocks = []
        for i in range(len(hidden_dims)):
            blocks.append(RNNBlock(d_model, hidden_dims[i], d_model))
        self.rnn = nn.ModuleList(blocks)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        hidden = None
        for layer in self.rnn:
            out = torch.zeros(batch_size, seq_len, layer.out_dim, device = CM().get('device'))
            for i in range(seq_len):
                i_out, hidden = layer(x[:, i], hidden)
                out[:, i, :] = i_out
            x = out
            hidden = None
            # out, hidden = self.rnn(x[:, i], hidden)
        return out

class Decoder(nn.Module):
    def __init__(self, d_model: int, hidden_dims: list[int]):
        super().__init__()
        blocks = []
        for i in range(len(hidden_dims)):
            blocks.append(RNNBlock(d_model, hidden_dims[i], d_model))
        self.rnn = nn.ModuleList(blocks)
    
    def forward(self, x, encoder_output):
        batch_size, seq_len, _ = x.shape
        hidden = encoder_output[:, 0, :]
        for layer in self.rnn:
            out = torch.zeros(batch_size, seq_len, layer.out_dim, device = CM().get('device'))
            for i in range(seq_len):
                i_out, hidden = layer(x[:, i], hidden)
                out[:, i, :] = i_out
            x = out
            hidden = None
        return out

class TrainableRNN(nn.Module):
    def __init__(self, d_model: int, encoder_dims: dict, decoder_dims: dict):
        super().__init__()
        self.encoder_projection = nn.Linear(encoder_dims['in'], d_model)
        self.encoder = Encoder(d_model, encoder_dims['hidden'])
        self.thicknesses_decoder_projection = nn.Linear(decoder_dims['in'], d_model)
        self.materials_decoder_projection = nn.Linear(decoder_dims['in'], d_model)
        self.thicknesses_decoder = Decoder(d_model, decoder_dims['hidden'])
        self.materials_decoder = Decoder(d_model, decoder_dims['hidden'])
        self.material_out = nn.Sequential(
            nn.Linear(d_model, decoder_dims['material_out']),
            nn.ReLU()
        )
        self.thickness_out = nn.Sequential(
            nn.Linear(d_model, decoder_dims['thickness_out']),
            ThicknessPostProcess(decoder_dims['seq_len'] - 1)
        )

class RNN(BaseSequentialModel):
    """
    Trainable prediction model using an RNN as base.

    Attributes:
        model: Instance of TrainableRNN.
    """
    def __init__(self):
        """Initialise an MLP instance."""
        super().__init__()

    def build_model(self):
        d_model = CM().get('rnn.d_model')
        encoder_dims = {
            'in': self.src_dim,
            'hidden': CM().get('rnn.encoder_dims'),
        }
        decoder_dims = {
            'in': 1,
            'hidden': [d_model] + CM().get('rnn.decoder_dims'),
            'thickness_out': self.out_dims['thickness'],
            'material_out': self.out_dims['material'],
            'seq_len': self.out_dims['seq_len']
        }
        return TrainableRNN(d_model, encoder_dims, decoder_dims).to(CM().get('device'))

    def project_encoder(self, src):
        return self.model.encoder_projection(src)

    def project_decoder(self, tgt):
        thicknesses, materials = tgt.chunk(2, -1)
        projected_thicknesses = self.model.thicknesses_decoder_projection(thicknesses)
        projected_materials = self.model.materials_decoder_projection(materials)
        return projected_thicknesses, projected_materials

    def encode(self, src):
        return self.model.encoder(src)[:, -1:, :]

    def decode(self, encoder_output, tgt_thicknesses, tgt_materials, mask = None):
        thicknesses = self.model.thicknesses_decoder(tgt_thicknesses, encoder_output)
        materials = self.model.materials_decoder(tgt_materials, encoder_output)
        return thicknesses, materials

    def output_thicknesses(self, decoder_output):
        return self.model.thickness_out(decoder_output)

    def output_materials(self, decoder_output):
        return self.model.material_out(decoder_output)

    def make_tgt_mask(self, tgt):
        return None

    def get_architecture_name(self):
        """
        Return name of model architecture.
        """
        return "rnn"

    def get_shared_params(self):
        params = []
        for param in self.model.encoder_projection.parameters():
            params.append(param)
        for param in self.model.encoder.parameters():
            params.append(param)
        for param in self.model.thicknesses_decoder_projection.parameters():
            params.append(param)
        for param in self.model.materials_decoder_projection.parameters():
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