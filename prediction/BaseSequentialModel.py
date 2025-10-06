from abc import ABC, abstractmethod
import torch
import sys
sys.path.append(sys.path[0] + '/..')
from prediction.BaseTrainableModel import BaseTrainableModel
from utils.ConfigManager import ConfigManager as CM

class BaseSequentialModel(BaseTrainableModel, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def project_encoder(self, src):
        """Project source into encoder dimension"""
        pass

    @abstractmethod
    def project_decoder(self, tgt):
        """Project target into decoder dimension"""
        pass

    @abstractmethod
    def encode(self, src):
        """Run source through encoder"""
        pass

    @abstractmethod
    def decode(self, encoder_output, tgt, mask = None):
        """Run target through decoder"""
        pass

    @abstractmethod
    def output_thicknesses(self, decoder_output):
        """Transform decoder output to thicknesses prediction"""
        pass

    @abstractmethod
    def output_materials(self, decoder_output):
        """Transform decoder output to materials prediction"""
        pass

    @abstractmethod
    def make_tgt_mask(self, tgt):
        """Create target mask"""
        pass

    def get_model_output(self, src, tgt = None):
        """
        Get output of the model for given input.

        Args:
            src: Input data.
            tgt: Target data.

        Returns:
            Output of the model.
        """
        src = self.project_encoder(src)
        encoder_output = self.encode(src)

        if tgt is not None and len(tgt.shape) != 1:
            # in training mode, target is specified
            # in training mode explicit leg, target is dummy data (len(shape) == 1) and should be ignored -> move to inference block
            mask = self.make_tgt_mask(tgt[:, :-1, :])
            batch_size = tgt.shape[0]
            tgt = self.project_decoder(tgt[:, :-1, :])
            bos = self.get_bos()[None].repeat(batch_size, 1, 1)
            bos_probability = self.indices_to_probs(bos)
            bos_thickness = torch.ones((batch_size, 1, 1), device = CM().get('device')) * CM().get('thicknesses_max') / 2
            decoder_output = self.decode(encoder_output, tgt, mask)
            out_thicknesses = self.output_thicknesses(decoder_output)
            out_materials = self.output_materials(decoder_output)
            connected_thicknesses = torch.cat([bos_thickness, out_thicknesses], dim = 1)
            connected_materials = torch.cat([bos_probability, out_materials], dim = 1)
            return connected_thicknesses, connected_materials
        else:
            batch_size = src.shape[0]
            bos_thickness = torch.ones((batch_size, 1, 1), device = CM().get('device')) * CM().get('thicknesses_max') / 2
            bos = self.get_bos()[None].repeat(batch_size, 1, 1)
            bos_probability = self.indices_to_probs(bos)
            tgt = torch.cat([bos_thickness, bos], dim = -1) # (batch, |coating| = 1, 2)
            tgt = self.project_decoder(tgt)
            while tgt.shape[1] < self.tgt_seq_len:
                decoder_output = self.decode(encoder_output, tgt)
                next = decoder_output[:, -1:] # take only the last item but keep dimension
                tgt = torch.cat([tgt, next], dim = 1)
            out_thicknesses = self.output_thicknesses(tgt[:, 1:, :])
            out_materials = self.output_materials(tgt[:, 1:, :])
            connected_thicknesses = torch.cat([bos_thickness, out_thicknesses], dim = 1)
            connected_materials = torch.cat([bos_probability, out_materials], dim = 1)
            return connected_thicknesses, connected_materials