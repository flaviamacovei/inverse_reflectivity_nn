import torch
import torch.optim as optim
import sys
sys.path.append(sys.path[0] + '/..')
from prediction.BasePredictionEngine import BasePredictionEngine
from data.values.ReflectivePropsPattern import ReflectivePropsPattern
from data.values.Coating import Coating
from forward.forward_tmm import coating_to_reflective_props
from evaluation.loss import compute_loss


class GradientPredictor(BasePredictionEngine):
    def __init__(self, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        thicknesses_initialisation = torch.randn((1, self.num_layers))
        thicknesses_initialisation.requires_grad_()
        self.thicknesses = thicknesses_initialisation
        self.optimiser = optim.AdamW([self.thicknesses], lr = 0.01)

    def predict(self, pattern: ReflectivePropsPattern):
        coating = Coating(self.thicknesses)
        start_wl = pattern.get_start_wl()
        end_wl = pattern.get_end_wl()
        steps = pattern.get_lower_bound().shape[0]
        epochs = 200
        for epoch in range(epochs):
            preds = coating_to_reflective_props(coating, start_wl, end_wl, steps)
            loss = compute_loss(preds, pattern)
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()
            if epoch % 10 == 0:
                print(f"loss in epoch {epoch + 1}: {loss}")
        return Coating(self.thicknesses.detach())
