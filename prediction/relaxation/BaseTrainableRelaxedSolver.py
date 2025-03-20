from abc import ABC, abstractmethod
import torch
import wandb
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
        self.compute_loss = self.loss_functions.get(CM().get('training.guidance'))

        self.model = None
        self.optimiser = None

    def train(self):
        if CM().get('wandb_log'):
            wandb.init(
                project=CM().get('wandb.project'),
                config=CM().get('wandb.config')
            )
        print("Training model...")
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
        print("Training complete.")
        model_filename = get_unique_filename(f"out/models/model_{CM().get('training.guidance')}_{'switch' if CM().get('training.dataset_switching') else 'no-switch'}_{CM().get('wavelengths').size()[0]}.pt")
        print(f"Saving model to {model_filename}")
        torch.save(self.model, model_filename)
        if CM().get('wandb_log'):
            wandb.log({"saved under": model_filename})

    def solve(self, target: ReflectivePropsPattern):
        self.set_to_eval()
        print(f"target shape: {target.get_lower_bound().shape}")
        model_input = torch.cat((target.get_lower_bound(), target.get_upper_bound()), dim = 1)
        return self.scaled_forward(model_input)

    def scaled_forward(self, target: torch.Tensor):
        coating_props = self.model(target.float())
        coating_props = coating_props.reshape((coating_props.shape[0], CM().get('layers.max'), CM().get('material_embedding.dim') + 1))
        # TODO: add scaling
        return Coating.from_encoding(coating_props)

    def compute_loss_guided(self, batch: (torch.Tensor, torch.Tensor)):
        features, labels = batch
        features = features.float().to(CM().get('device'))
        labels = labels.float().to(CM().get('device'))
        coating = self.scaled_forward(features)
        preds = coating.get_encoding()
        return torch.sum((preds - labels)**2)**0.5

    def compute_loss_free(self, batch: torch.Tensor):
        features = batch[0].float().to(CM().get('device'))
        print(f"features: {features.shape}")
        lower_bound, upper_bound = features.chunk(2, dim=1)
        pattern = ReflectivePropsPattern(lower_bound, upper_bound)
        print(f"pattern lower bound shape: {pattern.get_lower_bound().shape}")
        coating = self.scaled_forward(features)
        print(f"coating batch size: {coating.get_encoding().shape[0]}")
        preds = coating_to_reflective_props(coating)
        return match(preds, pattern)

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