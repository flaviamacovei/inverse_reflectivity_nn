import torch
import torch.nn as nn
import sys
sys.path.append(sys.path[0] + '/..')
from utils.ConfigManager import ConfigManager as CM
from prediction.BaseTrainableModel import BaseTrainableModel, ThicknessPostProcess
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
        hidden = encoder_output
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
        self.decoder_projection = nn.Linear(decoder_dims['in'], d_model)
        self.decoder = Decoder(d_model, decoder_dims['hidden'])
        self.material_out = nn.Linear(d_model, decoder_dims['material_out'])
        self.thickness_out = nn.Sequential(
            nn.Linear(d_model, decoder_dims['thickness_out']),
            ThicknessPostProcess(decoder_dims['seq_len'])
        )

    def project_encoder(self, src):
        return self.encoder_projection(src)

    def encode(self, src):
        return self.encoder(src)[:, -1, :]

    def project_decoder(self, tgt):
        return self.decoder_projection(tgt)

    def decode(self, encoder_output, tgt):
        return self.decoder(tgt, encoder_output)

    def project_thicknesses(self, thicknesses):
        return self.thickness_out(thicknesses)

    def project_materials(self, materials):
        return self.material_out(materials)

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
        d_model = CM().get('rnn.d_model')
        encoder_dims = {
            'in': self.src_dim,
            'hidden': CM().get('rnn.encoder_dims'),
        }
        decoder_dims = {
            'in': self.tgt_dim,
            'hidden': CM().get('rnn.decoder_dims'),
            'thickness_out': self.out_dims['thickness'],
            'material_out': self.out_dims['material'],
            'seq_len': self.out_dims['seq_len']
        }
        return TrainableRNN(d_model, encoder_dims, decoder_dims).to(CM().get('device'))

    def get_model_output(self, src, tgt = None):
        """
        Get output of the model for given input.

        Args:
            src: Input data.
            tgt: Target data.

        Returns:
            Output of the model.
        """
        projected_input = self.model.project_encoder(src) # (batch_size, src_seq_len, d_model)
        encoder_output = self.model.encode(projected_input) # (batch_size, d_model)
        if tgt is not None and len(tgt.shape) != 1:
            # in training mode, target is specified
            # in training mode explicit leg, target is dummy data (len(shape) == 1) and should be ignored -> move to inference block
            projected_tgt = self.model.project_decoder(tgt) # (batch_size, tgt_seq_len, d_model)
            decoder_output = self.model.decode(encoder_output, projected_tgt) # (batch_size, tgt_seq_len, d_model)
            projected_thicknesses = self.model.project_thicknesses(decoder_output) # (batch_size, tgt_seq_len, 1)
            projected_materials = self.model.project_materials(decoder_output) # (batch_size, tgt_seq_len, vocab_size)
            return torch.cat([projected_thicknesses, projected_materials], dim = -1)
        else:
            # in inference mode, target is not specified
            thickness = torch.ones((1, 1, 1)).to(CM().get('device'))
            bos = self.get_bos()
            tgt = torch.cat([thickness, bos[None]], dim = -1).repeat(encoder_output.shape[0], 1, 1) # (batch_size, 1, 2)
            tgt = self.model.project_decoder(tgt) # (batch_size, 1, d_model)
            while tgt.shape[1] < self.tgt_seq_len:
                decoder_output = self.model.decode(encoder_output, tgt) # (batch_size, running_seq_len, d_model)
                next = decoder_output[:, -1:] # take only last item but keep dimension
                tgt = torch.cat([tgt, next], dim = 1) # (batch_size, running_seq_len, d_model)
            projected_thicknesses = self.model.project_thicknesses(tgt) # (batch_size, tgt_seq_len, 1)
            projected_materials = self.model.project_materials(tgt) # (batch_size, tgt_seq_len, vocab_size)
            return torch.cat([projected_thicknesses, projected_materials], dim = -1)

    def scale_gradients(self):
        if self.guidance == "free":
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

    def get_architecture_name(self):
        """
        Return name of model architecture.
        """
        return "rnn"
