from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.init as init
import wandb
import os
from datetime import datetime
import sys
sys.path.append(sys.path[0] + '/..')
from data.dataloaders.BaseDataloader import BaseDataloader
from prediction.BaseModel import BaseModel
from data.values.ReflectivePropsPattern import ReflectivePropsPattern
from data.values.ReflectivePropsValue import ReflectivePropsValue
from data.values.Coating import Coating
from utils.os_utils import get_unique_filename
from forward.forward_tmm import coating_to_reflective_props
from evaluation.loss import match
from utils.ConfigManager import ConfigManager as CM
from utils.os_utils import short_hash
from data.dataloaders.DynamicDataloader import DynamicDataloader
from ui.visualise import visualise

class BaseTrainableModel(BaseModel, ABC):
    """
    Abstract base class for trainable prediction models.

    This class provides a common interface for training prediction models.
    It is the base from which all trainable prediction models should inherit.

    Attributes:
        model: The model to be trained. Must extend nn.Module.
        dataloader: Dataloader to be used for training.
        optimiser: Optimiser to be used for training.
        compute_loss: Loss function to be used for training. This implements either the free or the guided loss
        current_leg: Current leg of the guidance schedule.

    Methods:
        get_current_leg: Return current leg of the guidance schedule.
        update_leg: Update leg to next leg of the guidance schedule.
        train: Train the model.
        initialise_weights: Initialise weights of the trainable model.
        init_dataloader: Initialise dataloader.
        predict: Predict a coating given a reflective properties pattern object.
        compute_loss_guided: Compute guided loss of a batch of data. Guided loss is the L2 loss between the predicted coating and the ground truth coating.
        compute_loss_free: Compute free loss of a batch of data. Free loss is the L2 loss between the predicted reflective properties value and the ground truth reflective properties pattern.
        load: Load the model from a file.
        set_to_train: Set trainable model to training mode.
        set_to_eval: Set trainable model to evaluation mode.
        scale_gradients: Scale gradients of trainable model. Must be implemented by subclasses.
    """

    def __init__(self, model: nn.Module):
        """
        Initialise a BaseTrainableModel instance.

        Args:
            model: The model to be trained. Must extend nn.Module.
        """
        super().__init__()
        self.model = model
        self.init_dataloader()
        # self.optimiser = torch.optim.Adam(self.model.parameters(), lr=CM().get('training.learning_rate'))

        # loss function depends on guidance of current leg
        self.loss_functions = {
            "free": self.compute_loss_free,
            "guided": self.compute_loss_guided
        }
        # learning rate (and by extension optimiser) depends on guidance of current leg
        self.optimisers = {
            "free": torch.optim.Adam(self.model.parameters(), lr=CM().get('training.learning_rate') / 200),
            "guided": torch.optim.Adam(self.model.parameters(), lr=CM().get('training.learning_rate'))
        }
        self.compute_loss = None
        self.optimiser = None
        self.current_leg = -1

    def get_current_leg(self, epoch):
        """Return current leg of the guidance schedule."""
        # calculate what percent of training is done
        percent_done = epoch / CM().get('training.num_epochs')
        for leg in range(CM().get('training.num_legs')):
            # find corresponding leg by iteratively subtracting percentages of past legs
            if percent_done < CM().get(f'training.guidance_schedule.{leg}.percent'):
                return leg
            else:
                percent_done -= CM().get(f'training.guidance_schedule.{leg}.percent')

    def update_leg(self, epoch):
        """
        Update leg to next leg of the guidance schedule.

        Args:
            epoch: Current epoch.
        """
        epoch_leg = self.get_current_leg(epoch)
        if epoch_leg != self.current_leg:
            # only update attributes if leg has changed
            self.current_leg = epoch_leg
            guidance = CM().get(f'training.guidance_schedule.{self.current_leg}.guidance')
            density = CM().get(f'training.guidance_schedule.{self.current_leg}.density')
            print(
                f"{'-' * 50}\nOn leg {guidance}-{density}")
            # update attributes: loss function, optimiser, dataloader
            self.compute_loss = self.loss_functions[guidance]
            self.optimiser = self.optimisers[guidance]
            self.dataloader.load_leg(self.current_leg)

    def train(self):
        """Train the model."""
        assert self.dataloader is not None, "No dataloader provided, model can only be used in evaluation mode."
        print("Training model...")
        self.model.apply(self.initialise_weights)
        self.set_to_train()
        loss_scale = None
        checkpoint = None
        for epoch in range(CM().get('training.num_epochs')):
            self.update_leg(epoch)
            self.optimiser.zero_grad()

            epoch_loss = torch.tensor(0.0, device=CM().get('device'))
            # training loop
            for batch in self.dataloader:
                loss = self.compute_loss(batch)
                epoch_loss += loss

                if CM().get('wandb.log'):
                    if not loss_scale:
                        loss_scale = loss
                    wandb.log({"loss": loss.item() / loss_scale})
                loss.backward()
                self.scale_gradients()
                self.optimiser.step()
            if epoch % max(1, CM().get('training.num_epochs') / 10) == 0:
                if CM().get('training.save_model'):
                    # logging at every 10% of training
                    new_checkpoint = get_unique_filename(f"out/models/checkpoint_{short_hash(self.model) + short_hash(epoch)}.pt")
                    # always save latest checkpoint
                    torch.save(self.model, new_checkpoint)
                    if checkpoint:
                        os.remove(checkpoint)
                    checkpoint = new_checkpoint
                    current_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                    print(f"Checkpoint at {current_time}: {checkpoint}")
                print(f"Loss in epoch {epoch + 1}: {epoch_loss.item()}")
                # visualise first item of batch
                features = batch[0].float().to(CM().get('device'))
                lower_bound, upper_bound = features.chunk(2, dim=1)
                refs = ReflectivePropsPattern(lower_bound, upper_bound)
                encoding = self.model(features).reshape(
                    (features.shape[0], (CM().get('layers.max') + 2), CM().get('material_embedding.dim') + 1))
                coating = Coating(encoding)
                preds = coating_to_reflective_props(coating)
                visualise(preds, refs, f"from_training_epoch_{epoch}")

        print("Training complete.")
        # save final trained model
        if CM().get('training.save_model'):
            model_filename = get_unique_filename(f"out/models/model_{short_hash(self.model)}.pt")
            print(f"Saving model to {model_filename}")
            torch.save(self.model, model_filename)
            if CM().get('wandb.log'):
                wandb.log({"saved under": model_filename})
        if checkpoint:
            os.remove(checkpoint)

    def initialise_weights(self, model: nn.Module):
        """Initialise model linear weights using Kaiming normal initialisation."""
        # TODO: add itialisation for other layer types
        if isinstance(model, nn.Linear):
            init.kaiming_normal_(model.weight, nonlinearity = 'relu')
            if model.bias is not None:
                init.zeros_(model.bias)

    def init_dataloader(self):
        """Initialise dataloader for first leg of guidance schedule."""
        self.dataloader = DynamicDataloader(batch_size=CM().get('training.batch_size'), shuffle=True)
        try:
            self.dataloader.load_leg(0)
        except FileNotFoundError:
            print("Dataset in current configuration not found. Please run generate_dataset.py first.")
            return

    def predict(self, target: ReflectivePropsPattern):
        """
        Predict a coating given a reflective properties pattern object.

        Args:
            target: Reflective properties pattern for which to perform prediction.
        """
        self.set_to_eval()
        # convert input to shape expected by model
        model_input = torch.cat((target.get_lower_bound(), target.get_upper_bound()), dim = 1)
        # predict encoding of coating
        encoding = self.model(model_input.float()).reshape((model_input.shape[0], (CM().get('layers.max') + 2), CM().get('material_embedding.dim') + 1))
        return Coating(encoding)

    def compute_loss_guided(self, batch: (torch.Tensor, torch.Tensor)):
        """
        Compute guided loss of a batch of data.

        Guided loss is the L2 loss between the predicted coating and the ground truth coating.

        Args:
            batch: Tuple of features and labels for which to compute loss.
        """
        features, labels = batch
        # features are reflective properties converted to model input shape
        features = features.float().to(CM().get('device'))
        # labels are encodings of ground truth coatings
        labels = labels.float().to(CM().get('device'))
        preds = self.get_model_output(features, labels, 'guided')
        preds = preds.reshape((features.shape[0], (CM().get('layers.max') + 2), CM().get('material_embedding.dim') + 1))
        return torch.sum((preds - labels)**2)**0.5

    def compute_loss_free(self, batch: torch.Tensor):
        """
        Compute free loss of a batch of data.

        Free loss is the L2 loss between the predicted reflective properties value and the ground truth reflective properties pattern.

        Args:
            batch: Tuple of features and labels for which to compute loss.
        """
        # reflective properties are simultaneously model input and ground truth
        features = batch[0].float().to(CM().get('device'))
        lower_bound, upper_bound = features.chunk(2, dim=1)
        # create reflective properties pattern for match operation
        pattern = ReflectivePropsPattern(lower_bound, upper_bound)
        encoding = self.get_model_output(features, guidance = 'free')
        encoding = encoding.reshape((features.shape[0], (CM().get('layers.max') + 2), CM().get('material_embedding.dim') + 1))
        coating = Coating(encoding)
        # convert predicted coating to reflective properties value
        preds = coating_to_reflective_props(coating)
        return match(preds, pattern)

    @abstractmethod
    def get_model_output(self, src, tgt = None, guidance = 'free'):
        """
        Get output of the model for given input. Must be implemented by subclasses.

        Inputs can be specified by src only or src and tgt.

        Args:
            src: Model input.
            tgt: Model target.
            guidance: Guidance type. Accepted values: "free" and "guided".

        Returns:
            Output of the model.
        """
        pass

    def load(self, filename: str):
        """Load the model from a file."""
        self.model = torch.load(filename)
        self.model.to(CM().get('device'))

    def set_to_train(self):
        """Set trainable model to training mode."""
        self.model.train()

    def set_to_eval(self):
        """Set trainable model to evaluation mode."""
        self.model.eval()

    @abstractmethod
    def scale_gradients(self):
        """Scale gradients of trainable model. Must be implemented by subclasses."""
        if CM().get(f'training.guidance_schedule.{self.current_leg}.guidance') == "free":
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

