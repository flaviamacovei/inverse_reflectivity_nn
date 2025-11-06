from abc import ABC, abstractmethod
import torch
import sys
sys.path.append(sys.path[0] + '/..')
from prediction.BaseTrainableModel import BaseTrainableModel
from utils.ConfigManager import ConfigManager as CM
from utils.math_utils import make_arange

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
        dim = torch.tensor(x.shape).ne(torch.tensor(idx.shape)).int().argmax() # dimension along which to remove is dimensions which differs between x and idx
        subtrahend = torch.zeros(x.shape, device=CM().get('device'))
        arange = make_arange(x.shape, dim).to(CM().get('device'))
        mask = arange.eq(idx)
        subtrahend[mask] = torch.inf
        return x - subtrahend

    def expand_sequences(self, sequences, out_thicknesses, out_logits, score, beam_size: int = 1):
        batch_size, num_candidates, seq_len, vocab_size = out_logits.shape
        if sequences.shape[1] != beam_size:
            sequences = sequences.repeat(1, beam_size, 1, 1)
        out_logits = out_logits.transpose(-1, -2).reshape(batch_size, -1, seq_len)
        expanded_score = score[:, :, None].expand(-1, -1, vocab_size).reshape(out_logits.shape)
        _, idx = torch.sort(out_logits + expanded_score, dim = 1, descending = True) # if I also save the values here I don't need the arange below but that might not be differentiable so let's see
        material_idxs = idx[:, :beam_size]
        predecessor_idxs = material_idxs // self.tgt_vocab_size # beams from which new expansions originate
        material_idxs_norm = material_idxs % self.tgt_vocab_size # bring indices back to vocab range

        # calculate new score
        arange = make_arange(out_logits.shape, 1).to(CM().get('device'))[:, None]
        mask = arange.eq(material_idxs[:, :, None]).int()
        material_probs = out_logits[:, None].expand(-1, beam_size, -1, -1) * mask
        material_probs = material_probs.sum(dim = 2)

        # filter origin beam
        # dim 1: possible predecessors
        # dim 2: possible beams
        arange = make_arange(sequences.shape, 1).to(CM().get('device'))[:, None].repeat(1, beam_size, 1, 1, 1)
        mask = arange.eq(predecessor_idxs[:, :, None, None]).int()
        predecessors = sequences[:, None].repeat(1, beam_size, 1, 1, 1) * mask
        predecessors = predecessors.sum(dim = 2)
        predecessors_score = score[:, None].repeat(1, beam_size, 1) * mask[:, :, :, 0, 0]
        predecessors_score = predecessors_score.sum(dim = 2)
        new_score = material_probs[:, :, -1] + predecessors_score

        new_items = torch.cat([out_thicknesses.expand(-1, beam_size, -1, -1), material_idxs_norm[:, :, None]], dim = -1)
        new_sequences = torch.cat([predecessors, new_items], dim = 2)

        return new_sequences, new_score


    def beam_search(self, encoder_output, tgt, beam_size: int = 1):
        # dimension 2 reserved for beam expansion
        candidates = tgt[:, None]
        candidate_scores = torch.ones(tgt.shape[0], 1, device = CM().get('device'))
        max_seq_len = self.tgt_seq_len
        while candidates.shape[2] < max_seq_len:
            batch_size, num_candidates, seq_len, embed_dim = candidates.shape
            # merge beam dimension with batch dimension
            tgt = candidates.reshape(batch_size * num_candidates, seq_len, embed_dim)
            mask = self.make_tgt_mask(tgt)
            tgt = self.project_decoder(tgt)
            decoded_thicknesses, decoded_materials = self.decode(encoder_output[:, None].repeat(1, num_candidates, 1, 1).reshape(batch_size * num_candidates, encoder_output.shape[1], encoder_output.shape[2]), tgt, mask)
            decoded_thicknesses = decoded_thicknesses[:, -1:, :]
            decoded_materials = decoded_materials[:, -1:, :]
            out_thicknesses = self.output_thicknesses(decoded_thicknesses.repeat(1, self.tgt_seq_len - 1, 1))[:, :1, :]
            out_thicknesses = out_thicknesses.reshape(batch_size, num_candidates, 1, -1)
            out_logits = self.output_materials(decoded_materials)
            out_logits = out_logits.reshape(batch_size, num_candidates, 1, -1)
            candidates, candidate_scores = self.expand_sequences(candidates, out_thicknesses, out_logits, candidate_scores, beam_size) # reorder arguments

        _, idx = torch.sort(candidate_scores, dim = 1, descending = True)
        idx = idx[:, :1]
        arange = make_arange(candidates.shape, dim = 1).to(CM().get('device'))
        mask = arange.eq(idx[:, :, None, None].expand(arange.shape)).int()
        best_sequences = candidates * mask
        best_sequences = best_sequences.sum(dim = 1)
        out_thicknesses = best_sequences[:, :, :1]
        out_logits = self.indices_to_probs(best_sequences[:, :, 1:])
        # bring probabilities to log space
        out_logits = torch.exp(out_logits)
        return out_thicknesses, out_logits

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
        bos_thickness = self.thicknesses_mean.reshape(1, 1, 1).repeat(batch_size, 1, 1)
        bos = self.get_bos()[None].repeat(batch_size, 1, 1)
        bos_probability = self.indices_to_probs(bos)
        # bring to log space
        bos_probability = torch.exp(bos_probability)
        if tgt is not None and len(tgt.shape) != 1:
            # in training mode, target is specified
            # in training mode explicit leg, target is dummy data (len(shape) == 1) and should be ignored -> move to inference block
            mask = self.make_tgt_mask(tgt[:, :-1, :])
            tgt = self.project_decoder(tgt[:, :-1, :])
            decoded_thicknesses, decoded_materials = self.decode(encoder_output, tgt, mask)
            out_thicknesses = self.output_thicknesses(decoded_thicknesses)
            out_materials = self.output_materials(decoded_materials)
            out_thicknesses = torch.cat([bos_thickness, out_thicknesses], dim = 1)
            out_materials = torch.cat([bos_probability, out_materials], dim = 1)
        else:
            # inference mode
            tgt = torch.cat([bos_thickness, bos], dim=-1)  # (batch, |coating| = 1, 2)
            beam_size = CM().get('sequential.beam_size')
            out_thicknesses, out_materials = self.beam_search(encoder_output, tgt, beam_size)
        return out_thicknesses, out_materials