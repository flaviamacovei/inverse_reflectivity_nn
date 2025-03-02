from abc import ABC, abstractmethod
import torch
import sys
sys.path.append(sys.path[0] + '/..')
from data.dataloaders.BaseDataloader import BaseDataloader
from prediction.relaxation.BaseRelaxedSolver import BaseRelaxedSolver
from data.values.ReflectivePropsPattern import ReflectivePropsPattern
from data.values.ReflectivePropsValue import ReflectivePropsValue
from data.values.Coating import Coating
from utils.os_utils import get_unique_filename
from forward.forward_tmm import coating_to_reflective_props
from evaluation.loss import match
from utils.ConfigManager import ConfigManager as CM

class BaseTrainableRelaxedSolver(BaseRelaxedSolver, ABC):
    def __init__(self, dataloader: BaseDataloader):
        super().__init__()
        self.dataloader = dataloader
        self.SCALING_FACTOR_THICKNESSES = CM().get('scaling.thicknesses')
        self.SCALING_FACTOR_REFRACTIVE_INDICES = CM().get('scaling.refractive_indices')

        self.loss_functions = {
            "free": self.compute_loss_free,
            "guided": self.compute_loss_guided
        }
        self.compute_loss = self.loss_functions.get(CM().get('training.loss_function'))

        self.model = None
        self.optimiser = None

    def train(self):
        self.set_to_train()
        loss_scale = None
        for epoch in range(CM().get('training.num_epochs')):
            self.dataloader.next_epoch()
            self.optimiser.zero_grad()
            epoch_loss = torch.tensor(0.0, device=CM().get('device'))
            for batch in self.dataloader:

                loss = self.compute_loss(batch)
                epoch_loss += loss

                if CM().get('wandb_log'):
                    if not loss_scale:
                        loss_scale = loss
                    wandb.log({"loss": loss.item() / loss_scale})

                loss.backward()
                self.scale_gradients()

                self.optimiser.step()
            if epoch % 1 == 0:
                print(f"Loss in epoch {epoch + 1}: {epoch_loss.item()}")
        model_filename = f"out/models/model_{CM().get('training.loss_function')}_{'switch' if CM().get('training.dataset_switching') else 'no-switch'}_{CM().get('wavelengths').size()[0]}.pt"
        torch.save(self.model, get_unique_filename(model_filename))

    def solve(self, target: ReflectivePropsPattern):
        self.set_to_eval()
        model_input = torch.cat((target.get_lower_bound(), target.get_upper_bound()), dim = 1)
        return self.scaled_forward(model_input)

    def scaled_forward(self, target: torch.Tensor):
        coating_props = self.model(target)
        thicknesses, refractive_indices = coating_props.chunk(2, dim = 1)
        thicknesses = thicknesses / self.SCALING_FACTOR_THICKNESSES
        refractive_indices = refractive_indices / self.SCALING_FACTOR_REFRACTIVE_INDICES
        return Coating(thicknesses, refractive_indices)

    def compute_loss_guided(self, batch: (torch.Tensor, torch.Tensor)):
        pattern, labels = batch
        pattern = pattern.float().to(CM().get('device'))
        labels = labels.float().to(CM().get('device'))
        coating = self.scaled_forward(pattern)
        preds = torch.cat((coating.get_thicknesses(), coating.get_refractive_indices()), dim=1)
        return torch.sum((preds - labels)**2)**0.5

    def compute_loss_free(self, batch: torch.Tensor):
        pattern = batch[0].float().to(CM().get('device'))
        lower_bound, upper_bound = pattern.chunk(2, dim=1)
        refs_obj = ReflectivePropsPattern(lower_bound, upper_bound)
        coating = self.scaled_forward(pattern)
        preds = coating_to_reflective_props(coating)
        return match(preds, refs_obj)

    def initialise_opitimiser(self):
        if self.model is not None:
            self.optimiser = torch.optim.Adam(self.model.parameters(), lr=CM().get('training.learning_rate'))
        else:
            raise Exception("Initialise model before initialising optimiser")

    @abstractmethod
    def set_to_train(self):
        pass

    @abstractmethod
    def set_to_eval(self):
        pass

    @abstractmethod
    def scale_gradients(self):
        pass