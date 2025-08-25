from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.init as init
import wandb
import os
from datetime import datetime
import sys
import yaml
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
from data.material_embedding.EmbeddingManager import EmbeddingManager as EM

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
            "free": torch.optim.Adam(self.model.parameters(), lr=CM().get('training.free_learning_rate')),
            "guided": torch.optim.Adam(self.model.parameters(), lr=CM().get('training.guided_learning_rate'))
        }
        self.compute_loss = None
        self.optimiser = None
        self.current_leg = -1
        self.guidance = None
        self.checkpoint = None

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
            # update attributes: loss function, optimiser, dataloader,
            # first run data through model, then compute loss
            # shapes might be off (+ 1 due to changes in SegmentedDataset)
            self.compute_loss = self.loss_functions[guidance]
            self.optimiser = self.optimisers[guidance]
            self.dataloader.load_leg(self.current_leg)
            self.guidance = guidance

    def train(self):
        """Train the model."""
        assert self.dataloader is not None, "No dataloader provided, model can only be used in evaluation mode."
        print("Training model...")
        self.model.apply(self.initialise_weights)
        for epoch in range(CM().get('training.num_epochs')):
            self.model.train()
            self.update_leg(epoch)

            epoch_loss = torch.tensor(0.0, device=CM().get('device'))
            # training loop
            for batch in self.dataloader:
                self.optimiser.zero_grad()
                reflective_props_tensor, coating_encoding = batch
                reflective_props_tensor = reflective_props_tensor.to(CM().get('device'))
                coating_encoding = coating_encoding.to(CM().get('device'))
                # a batch consists of reflective properties and coating encodings
                # in free mode, reflective properties are simultaneously input and ground truth
                # in guided mode, coating encodings are ground truth
                labels = reflective_props_tensor if self.guidance == 'free' else coating_encoding
                # TODO: might need to add target too for transformer
                preds = self.get_model_output(reflective_props_tensor)
                preds = preds.reshape((reflective_props_tensor.shape[0], (CM().get('layers.max') + 2), CM().get('material_embedding.dim') + 1))
                loss = self.compute_loss(preds, labels)
                # print(f"loss: {loss}")
                epoch_loss += loss.item()

                if CM().get('wandb.log'):
                    wandb.log({"loss": loss.item()})

                loss.backward()

                # for name, param in self.model.named_parameters():
                #     if param.requires_grad:
                #         print(name, param.data)

                self.scale_gradients()
                self.optimiser.step()
            if epoch % max(1, CM().get('training.num_epochs') / 10) == 0:
                if CM().get('training.save_model'):
                    # logging at every 10% of training
                    self.update_checkpoint(epoch)
                print(f"Loss in epoch {epoch + 1}: {epoch_loss.item()}")
                # divide by dataset size instead of loss in epoch 0?
                self.visualise_epoch(epoch)
        print("Training complete.")
        # save final trained model
        if CM().get('training.save_model'):
            self.save_model()
        if self.checkpoint:
            os.remove(self.checkpoint)

    def visualise_epoch(self, epoch: int):
        # visualise first item of batch
        self.model.eval()
        reflective_props_tensor = self.dataloader[0][0][None]
        reflective_props_tensor = reflective_props_tensor.to(CM().get('device'))
        lower_bound, upper_bound = reflective_props_tensor.chunk(2, dim=1)
        refs = ReflectivePropsPattern(lower_bound, upper_bound)
        output = self.get_model_output(reflective_props_tensor)
        output = output.reshape(
            (reflective_props_tensor.shape[0], (CM().get('layers.max') + 2), CM().get('material_embedding.dim') + 1))
        coating = Coating(output)
        preds = coating_to_reflective_props(coating)
        print(coating.get_batch(0))
        visualise(preds, refs, f"from_training_epoch_{epoch}")

    def update_checkpoint(self, epoch: int):
        new_checkpoint = get_unique_filename(f"out/models/checkpoint_{short_hash(self.model) + short_hash(epoch)}.pt")
        # always save latest checkpoint
        torch.save(self.model, new_checkpoint)
        if self.checkpoint:
            os.remove(self.checkpoint)
        self.checkpoint = new_checkpoint
        current_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        print(f"Checkpoint at {current_time}: {self.checkpoint}")

    def save_model(self):
        MODEL_METADATA = "out/models/models_metadata.yaml"
        model_filename = get_unique_filename(f"out/models/model_{short_hash(self.model)}.pt")

        props_dict = {
            "architecture": CM().get('architecture'),
            "num_layers": CM().get('num_layers'),
            "min_wl": CM().get('wavelengths')[0].item(),
            "max_wl": CM().get('wavelengths')[-1].item(),
            "wl_step": len(CM().get('wavelengths')),
            "polarisation": CM().get('polarisation'),
            "materials_hash": EM().hash_materials(),
            "num_materials": len(CM().get('materials.thin_films')),
            "theta": CM().get('theta').item(),
            "air_pad": CM().get('air_pad'),
            "tolerance": CM().get('tolerance'),
            "num_points": CM().get('training.dataset_size'),
            "epochs": CM().get('training.num_epochs')
        }
        if not os.path.exists(MODEL_METADATA):
            # create file if it does not exist
            with open(MODEL_METADATA, "w") as f:
                yaml.dump({"models": [{**{'title': model_filename}, **{'properties': props_dict}}]}, f, sort_keys=False)
        else:
            with open(MODEL_METADATA, "r+") as f:
                content = yaml.safe_load(f)
                content['models'].append({**{'title': model_filename}, **{'properties': props_dict}})
                f.seek(0)
                yaml.dump(content, f, sort_keys=False, default_flow_style=False, indent=2)
        print(f"Saving model to {model_filename}")
        torch.save(self.model, model_filename)
        if CM().get('wandb.log'):
            wandb.log({"saved under": model_filename})

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
        self.model.eval()
        # convert input to shape expected by model
        model_input = torch.cat((target.get_lower_bound(), target.get_upper_bound()), dim = 1)
        # predict encoding of coating
        preds = self.get_model_output(model_input)
        preds = preds.reshape(
            (model_input.shape[0], (CM().get('layers.max') + 2), CM().get('material_embedding.dim') + 1))
        return Coating(preds)

    def compute_loss_guided(self, preds: torch.Tensor, labels: torch.Tensor):
        """
        Compute guided loss of a batch of data.

        Guided loss is the L2 loss between the predicted coating and the ground truth coating.

        Args:
            preds: Model output.
            labels: Ground truth coating encoding.
        """
        # features are reflective properties converted to model input shape
        return torch.sum((preds - labels) ** 2)

    def compute_loss_free(self, preds: torch.Tensor, labels: torch.Tensor):
        """
        Compute free loss of a batch of data.

        Free loss is the L2 loss between the predicted reflective properties value and the ground truth reflective properties pattern.

        Args:
            batch: Tuple of features and labels for which to compute loss.
        """
        lower_bound, upper_bound = labels.chunk(2, dim=1)
        # create reflective properties pattern for match operation
        pattern = ReflectivePropsPattern(lower_bound, upper_bound)
        coating = Coating(preds)
        # convert predicted coating to reflective properties value
        preds = coating_to_reflective_props(coating)
        return match(preds, pattern)

    @abstractmethod
    def get_model_output(self, src, tgt = None):
        """
        Get output of the model for given input. Must be implemented by subclasses.

        Inputs can be specified by src only or src and tgt.

        Args:
            src: Model input.
            tgt: Model target.

        Returns:
            Output of the model.
        """
        pass

    def load(self, filename: str, weights_only: bool = True):
        """Load the model from a file."""
        self.model = torch.load(filename, weights_only = weights_only)
        self.model.to(CM().get('device'))


    @abstractmethod
    def scale_gradients(self):
        """Scale gradients of trainable model. Must be implemented by subclasses."""

