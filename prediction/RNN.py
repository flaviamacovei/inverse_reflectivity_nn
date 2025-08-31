import torch
import torch.nn as nn
import sys
sys.path.append(sys.path[0] + '/..')
from utils.ConfigManager import ConfigManager as CM
from data.material_embedding.EmbeddingManager import EmbeddingManager as EM
from prediction.BaseTrainableModel import BaseTrainableModel

class RNNBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.input_to_hidden = nn.Linear(self.in_dim + self.hidden_dim, self.hidden_dim)
        self.input_to_output = nn.Linear(self.in_dim + self.hidden_dim, self.out_dim)

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
    def __init__(self):
        super().__init__()
        in_dim = 2 # lower and upper bound
        hidden_dim = CM().get('rnn.encoder_dim')
        out_dim = CM().get('rnn.decoder_dim')

        self.rnn = RNNBlock(in_dim, hidden_dim, out_dim)

    def forward(self, x):
        seq_len = x.shape[1]
        hidden = None
        out = None
        for i in range(seq_len):
            out, hidden = self.rnn(x[:, i], hidden)
        return out

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        in_dim = CM().get('material_embedding.dim') + 1
        hidden_dim = CM().get('rnn.decoder_dim')
        out_dim = CM().get('material_embedding.dim') + 1
        
        self.rnn = RNNBlock(in_dim, hidden_dim, out_dim)
    
    def forward(self, x, encoder_output):
        seq_len = x.shape[1]
        hidden = encoder_output
        sequence = None
        for i in range(seq_len):
            out, hidden = self.rnn(x[:, i], hidden)
            sequence = out if sequence is None else torch.cat([sequence, out], dim = 1)
        return sequence

class TrainableRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def encode(self, src):
        return self.encoder(src)

    def decode(self, encoder_output, tgt):
        return self.decoder(tgt, encoder_output)

class RNN(BaseTrainableModel):
    """
    Trainable prediction model using an RNN as base.

    Attributes:
        model: Instance of TrainableRNN.
    """
    def __init__(self):
        """Initialise an MLP instance."""
        super().__init__(TrainableRNN().to(CM().get('device')))

    def get_model_output(self, src, tgt = None):
        """
        Get output of the model for given input.

        Args:
            src: Input data.
            tgt: Target data.

        Returns:
            Output of the model.
        """
        lower_bound, upper_bound = torch.chunk(src, 2, 1)
        src = torch.stack([lower_bound, upper_bound], dim=-1)  # (batch, 2 * |wl|) --> (batch, |wl|, 2)
        encoder_output = self.model.encode(src)
        if tgt is not None and len(tgt.shape) != 1:
            # in training mode, target is specified
            # in training mode explicit leg, target is dummy data (len(shape) == 3) and should be ignored -> move to inference block
            decoder_output = self.model.decode(encoder_output, tgt)
            return decoder_output
        else:
            # in inference mode, target is not specified
            substrate = EM().get_material(CM().get('materials.substrate'))
            # beginning of any coating: thickness is 1.0, material is substrate
            thickness = torch.ones((src.shape[0], 1, 1), device=CM().get('device'))  # (batch, 1, |coating| = 1, 1)
            substrate_encoding = EM().encode([substrate])[:, None].repeat(src.shape[0], 1, 1)  # (batch, |coating| = 1, 1)
            tgt = torch.cat([thickness, substrate_encoding], dim=-1)  # (batch, |coating| = 1, tgt_embed_dim + 1)
            tgt_seq_len = CM().get('num_layers') + 2
            while tgt.shape[1] < tgt_seq_len:
                tgt = self.model.decode(encoder_output, tgt)
            return tgt

    def scale_gradients(self):
        if self.guidance == "free":
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

    def get_architecture_name(self):
        """
        Return name of model architecture.
        """
        return "rnn"