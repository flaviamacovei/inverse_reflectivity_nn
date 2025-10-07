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

    def remove(self, x: torch.Tensor, idx: torch.Tensor):
        subtrahend = torch.zeros(x.shape, device=CM().get('device'))
        arange = torch.arange(x.shape[-1])[None, None].repeat(x.shape[0], x.shape[1], 1)
        mask = arange.eq(idx)
        subtrahend[mask] = torch.inf
        return x - subtrahend

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

        batch_size = src.shape[0]
        bos_thickness = torch.ones((batch_size, 1, 1), device = CM().get('device')) * CM().get('thicknesses_max') / 2
        bos = self.get_bos()[None].repeat(batch_size, 1, 1)
        bos_probability = self.indices_to_probs(bos)
        if tgt is not None and len(tgt.shape) != 1:
            # in training mode, target is specified
            # in training mode explicit leg, target is dummy data (len(shape) == 1) and should be ignored -> move to inference block
            mask = self.make_tgt_mask(tgt[:, :-1, :])
            tgt = self.project_decoder(tgt[:, :-1, :])
            decoder_output = self.decode(encoder_output, tgt, mask)
            out_thicknesses = self.output_thicknesses(decoder_output)
            out_materials = self.output_materials(decoder_output)
            out_thicknesses = torch.cat([bos_thickness, out_thicknesses], dim=1)
            out_materials = torch.cat([bos_probability, out_materials], dim=1)
        else:
            # inference mode
            tgt = torch.cat([bos_thickness, bos], dim = -1) # (batch, |coating| = 1, 2)
            # list of dictionaries each with keys: sequence, score, pending
            candidates = [{
                'sequence': tgt,
                'score': 1,
                'pending': True
            }]
            beam_size = CM().get('sequential.beam_size')
            max_seq_len = self.tgt_seq_len - 1
            out_sequences = []
            while len(candidates) > 0:
                expansions = []
                for candidate in candidates:
                    candidate['pending'] = False
                    sequence = candidate['sequence']
                    score = candidate['score']
                    if sequence.shape[1] > max_seq_len or sequence[:, -1, 1].eq(self.get_eos()[None].repeat(batch_size, 1, 1)).all():
                        out_sequences.append(candidate)
                        continue
                    mask = self.make_tgt_mask(sequence)
                    tgt = self.project_decoder(sequence)
                    decoder_output = self.decode(encoder_output, tgt, mask)[:, -1:, :]
                    out_thicknesses = self.output_thicknesses(decoder_output.repeat(1, self.tgt_seq_len - 1, 1))[:, :1, :]
                    out_materials = self.output_materials(decoder_output)
                    for i in range(beam_size):
                        # for each of the top k candidates, get the material and its probability
                        material_prob, material_idx = torch.max(out_materials, dim = -1, keepdim = True)
                        new_item = torch.cat([out_thicknesses, material_idx], dim = -1)
                        new_sequence = torch.cat([sequence, new_item], dim = 1)
                        expansions.append({
                            'sequence': new_sequence,
                            # sum probabilities because they are in log space
                            'score': score + material_prob.sum(),
                            'pending': True
                        })
                        out_materials = self.remove(out_materials, material_idx)
                candidates = list(filter(lambda x: x['pending'], candidates))
                candidates.extend(expansions)
                candidates = sorted(candidates, key = lambda x: x['score'], reverse = True)
                candidates = candidates[:beam_size]
            out_sequences = sorted(out_sequences, key = lambda x: x['score'], reverse = True)
            best = out_sequences[0]
            out_thicknesses = best['sequence'][:, :, :1]
            out_materials = self.indices_to_probs(best['sequence'][:, :, 1:])
            # bring probabilities to log space
            out_materials[out_materials == 0] = -torch.inf
        return out_thicknesses, out_materials