from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import wandb
import os
from datetime import datetime
import sys
import yaml
import math
import itertools

sys.path.append(sys.path[0] + '/..')
from prediction.BaseModel import BaseModel
from data.values.ReflectivityPattern import ReflectivityPattern
from data.values.ReflectivityValue import ReflectivityValue
from data.values.Coating import Coating
from utils.os_utils import get_unique_filename
from forward.forward_tmm import coating_to_reflectivity
from evaluation.loss import match
from utils.ConfigManager import ConfigManager as CM
from utils.os_utils import short_hash
from data.dataloaders.DynamicDataloader import DynamicDataloader
from ui.visualise import visualise
from data.material_embedding.EmbeddingManager import EmbeddingManager as EM
from utils.data_utils import get_saved_model_path
from evaluation.model_eval import evaluate_model
from utils.math_utils import ArgMax

class ThicknessPostProcess(nn.Module):
    def __init__(self, dims: int):
        super().__init__()
        # self.norm = nn.BatchNorm1d(dims)
        self.net = nn.BatchNorm1d(dims)

    def forward(self, x):
        # return torch.exp(self.net(x))
        return self.net(x)

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
        current_leg: Current leg of the curriculum.

    Methods:
        get_current_leg: Return current leg of the curriculum.
        update_leg: Update leg to next leg of the curriculum.
        train: Train the model.
        initialise_weights: Initialise weights of the trainable model.
        init_dataloader: Initialise dataloader.
        predict: Predict a coating given a reflectivity pattern object.
        compute_loss_guided: Compute guided loss of a batch of data. Guided loss is the L2 loss between the predicted coating and the ground truth coating.
        compute_loss_free: Compute free loss of a batch of data. Free loss is the L2 loss between the predicted reflectivity value and the ground truth reflectivity pattern.
        load: Load the model from a file.
        scale_gradients: Scale gradients of trainable model. Must be implemented by subclasses.
    """

    def __init__(self):
        """
        Initialise a BaseTrainableModel instance.

        Args:
            model: The model to be trained. Must extend nn.Module.
        """
        super().__init__()

        self.init_dataloader()

        self.current_leg = -1
        self.checkpoint = None

        self.vocab = EM().get_refractive_indices_table()
        self.num_epochs = CM().get('training.num_epochs')
        self.epoch = 0

        self.model = self.build_model()
        learning_rate = CM().get('training.learning_rate.start')
        self.optimiser = torch.optim.Adam(self.model.parameters(), learning_rate)
        lr_end_factor = min(max(0, CM().get('training.learning_rate.stop') / learning_rate), 1)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimiser, start_factor = 1.0, end_factor = lr_end_factor, total_iters = self.num_epochs)

        self.early_stopping_counter = 0

        # normalisation values
        self.reflectivity_mean = self.reflectivity_std = self.thicknesses_mean = self.thicknesses_std = self.materials_mean = self.materials_std = None
        DynamicDataloader.register(self)

    @abstractmethod
    def get_shared_params(self):
        pass
    @abstractmethod
    def get_thicknesses_params(self):
        pass
    @abstractmethod
    def get_materials_params(self):
        pass

    def handle_new_data(self, dataset):
        # TODO: these are all 0 if epochs == 0
        # also maybe call it every time the dataloader is renewed? even during evaluation / testing?
        # plus we need a whole different approach for explicit
        data = dataset[0: len(dataset)]
        if len(data) == 2:
            reflectivity, coating = data
        else:
            reflectivity = data[0]
            # in test mode: dummy data
            coating = torch.zeros((len(reflectivity), 2))
        reflectivity = reflectivity.to(CM().get('device'))
        coating = coating.to(CM().get('device'))
        thicknesses, material_indices = coating.chunk(2, -1)
        self.reflectivity_mean = reflectivity.mean() if reflectivity.mean() != 0 else torch.tensor([0.5], device = CM().get('device'))
        self.reflectivity_std = reflectivity.std() if reflectivity.std() != 0 else torch.tensor([0.25], device = CM().get('device'))
        self.thicknesses_mean = thicknesses.mean() if thicknesses.mean() != 0 else torch.tensor([CM().get('thicknesses_max') / 2], device = CM().get('device'))
        self.thicknesses_std = thicknesses.std() if thicknesses.std() != 0 else torch.tensor([CM().get('thicknesses_max') / 4], device = CM().get('device'))
        self.materials_mean = material_indices.mean() if material_indices.mean() != 0 else torch.tensor([self.tgt_vocab_size / 2], device = CM().get('device'))
        self.materials_std = material_indices.std() if material_indices.std() != 0 else torch.tensor([self.tgt_vocab_size / 4], device = CM().get('device'))


    def normalise_reflectivity(self, reflectivity):
        return (reflectivity - self.reflectivity_mean) / self.reflectivity_std

    def normalise_thicknesses(self, thicknesses):
        return (thicknesses - self.thicknesses_mean) / self.thicknesses_std

    def normalise_materials(self, materials):
        return (materials - self.materials_mean) / self.materials_std

    def denormalise_reflectivity(self, reflectivity):
        return reflectivity * self.reflectivity_std + self.reflectivity_mean

    def denormalise_thicknesses(self, thicknesses):
        return thicknesses * self.thicknesses_std + self.thicknesses_mean

    def denormalise_materials(self, materials):
        return materials * self.materials_std + self.materials_mean


    @abstractmethod
    def build_model(self):
        """
        Returns a model that extends nn.Module based on the architecture of the subclass.
        """
        pass

    def get_current_leg(self, epoch):
        """Return current leg of the curriculum."""
        # calculate what percent of training is done
        percent_done = epoch / self.num_epochs
        for leg in range(CM().get('training.num_legs')):
            # find corresponding leg by iteratively subtracting percentages of past legs
            if percent_done < CM().get(f'training.curriculum.{leg}.percent'):
                return leg
            else:
                percent_done -= CM().get(f'training.curriculum.{leg}.percent')

    def update_leg(self, epoch):
        """
        Update leg to next leg of the curriculum.

        Args:
            epoch: Current epoch.
        """
        # update loss weights
        self.epoch = epoch
        # update leg
        epoch_leg = self.get_current_leg(epoch)
        if epoch_leg != self.current_leg:
            # only update attributes if leg has changed
            self.current_leg = epoch_leg
            self.density = CM().get(f'training.curriculum.{self.current_leg}.density')
            print(
                f"{'-' * 50}\nCurrent leg: {self.density}")
            self.dataloader.load_leg(self.current_leg)

    def add_scaled(self, grads, params, scale, weight = 1.0):
        for p, g in zip(params, grads):
            if g is None:
                continue
            g_scaled = g * (scale * weight)
            if p.grad is None:
                p.grad = g_scaled
            else:
                p.grad.add_(g_scaled)

    def train(self):
        """Train the model."""
        assert self.dataloader is not None, "No dataloader provided, model can only be used in evaluation mode."
        print("Training model...")
        self.model.apply(self.initialise_weights)
        for epoch in range(self.num_epochs):
            self.model.train()
            self.update_leg(epoch)

            epoch_loss = torch.tensor(0.0, device=CM().get('device'))
            # training loop
            for i, batch in enumerate(self.dataloader):
                self.optimiser.zero_grad()
                # a batch consists of reflectivity and coating encodings
                reflectivity, coating_encoding = batch
                reflectivity = reflectivity.to(CM().get('device'))
                reflectivity = torch.stack([reflectivity[:, :self.src_seq_len], reflectivity[:, self.src_seq_len:]], dim = 2)
                coating_encoding = coating_encoding.to(CM().get('device'))
                output = self.model_call(reflectivity, coating_encoding)

                free_loss = self.compute_loss_free(output, reflectivity)

                if self.density == 'explicit':
                    losses = [{'val': free_loss, 'infl': ['shared', 'thicknesses', 'materials'], 'weight': CM().get('training.free_factor')}]
                    # losses = [{'val': free_loss, 'infl': ['thicknesses'], 'weight': CM().get('training.free_factor')}]
                else:
                    thickness_loss, material_loss = self.compute_loss_guided(output, coating_encoding)
                    losses = [
                        {'val': material_loss, 'infl': ['materials'], 'weight': CM().get('training.material_factor')},
                        {'val': thickness_loss, 'infl': ['thicknesses'], 'weight': CM().get('training.thickness_factor')},
                        {'val': free_loss, 'infl': ['shared'], 'weight': CM().get('training.free_factor')},
                    ]

                self.compute_grads(losses)

                loss = sum(l['val'] for l in losses)
                epoch_loss += loss.item()

                if CM().get('wandb.log') or CM().get('wandb.sweep'):
                    wandb.log({
                        "loss": loss.item(),
                        "free_loss": free_loss.item(),
                        "guided_loss": material_loss.item() + thickness_loss.item() if self.density != 'explicit' else 0,
                    })

                # loss.backward()

                # self.scale_gradients()
                self.optimiser.step()

            epoch_loss /= len(self.dataloader)
            self.scheduler.step()
            if epoch % max(1, self.num_epochs / 20) == 0:
                if CM().get('training.save_model'):
                    # logging at every 5% of training
                    self.update_checkpoint(epoch)
                print(f"Loss in epoch {epoch + 1}: {epoch_loss.item()}")
                self.evaluate_epoch(epoch)
                if self.early_stopping(CM().get('training.early_stopping.threshold'), CM().get('training.early_stopping.patience')):
                    print(f"Early stopping at epoch {self.epoch}")
                    break
        print("Training complete.")
        # save final trained model
        if CM().get('training.save_model'):
            self.save_model()
        if self.checkpoint:
            os.remove(self.checkpoint)

    def grad_list(self, loss, params):
        return torch.autograd.grad(loss, params, retain_graph = True, create_graph = False, allow_unused = True)

    def grad_norm(self, grads):
        return torch.sqrt(sum((g.detach()**2).sum() for g in grads if g is not None) + 1e-12)

    def compute_grads(self, losses: list[dict]):
        params = {
            'shared': [p for p in self.get_shared_params() if p.requires_grad],
            'thicknesses': [p for p in self.get_thicknesses_params() if p.requires_grad],
            'materials': [p for p in self.get_materials_params() if p.requires_grad],
        }

        for loss in losses:
            grads = dict()
            for influence in loss['infl']:
                grads[influence] = self.grad_list(loss['val'], params[influence])
            norm = self.grad_norm(list(itertools.chain.from_iterable(grads.values())))
            scale = 1.0 / norm
            for influence, grad in grads.items():
                self.add_scaled(grad, params[influence], scale, loss['weight'])

    def get_count(self, material_logits):
        materials = self.logits_to_indices(material_logits)[:, :, 0]
        mode, _ = torch.mode(materials, dim = -1, keepdim = True)
        return materials.eq(mode).sum(dim = -1)

    def early_stopping(self, t = 0.6, patience = 3):
        """
        Return true if prediction contains t% more occurrences of the same material per data point than references.
        """
        assert t >= 0 and t <= 1, f"Cutoff t must lie between 0 and 1 found {t}."
        self.model.eval()
        reflectivity, coating_encoding = self.dataloader[:min(CM().get('training.dataset_size'), 100)]
        reflectivity = reflectivity.to(CM().get('device'))
        if self.density == 'explicit':
            coating_encoding = None
        else:
            coating_encoding = coating_encoding.to(CM().get('device'))
        reflectivity = torch.stack([reflectivity[:, :self.src_seq_len], reflectivity[:, self.src_seq_len:]], dim=2)
        with torch.no_grad():
            thicknesses, logits = self.model_call(reflectivity, coating_encoding)
        preds_count = self.get_count(logits)
        if coating_encoding is None:
            # no reference proviced: own entropy
            max_count = logits.shape[1]
            stop = (preds_count / max_count).ge(t).all()
        else:
            # reference provided: compare entropy with reference
            refs_count = self.get_count(self.indices_to_probs(coating_encoding[:, :, 1:]))
            stop = preds_count.ge(refs_count * (1 + t)).all()
        if stop:
            # increment patience counter
            self.early_stopping_counter += 1
            if self.early_stopping_counter >= patience:
                return True
        else:
            # reset patience counter
            self.early_stopping_counter = 0
        return False

    def get_bos(self):
        return EM().get_substrate_index()[None]
        substrate = EM().get_substrate()
        bos = EM().materials_to_indices([[substrate]]) # (batch, |coating| = 1, 1)
        return bos

    def get_eos(self):
        return EM().get_air_index()[None]
        air = EM().get_air()
        eos = EM().materials_to_indices([[air]]) # (batch, |coating| = 1, 1)
        return eos

    def indices_to_probs(self, material_indices: torch.Tensor):
        long_indices = material_indices[:, :, 0].to(torch.long)
        return F.one_hot(long_indices, len(self.vocab)).to(torch.float)

    def logits_to_indices(self, logits: torch.Tensor):
        softmax_probabilities = F.softmax(logits, dim = -1)
        max_indices = ArgMax.apply(softmax_probabilities, -1, True)
        return max_indices

    def mask_logits(self, logits: torch.Tensor):
        seq_len = logits.shape[1]
        substrate = self.get_bos()
        air = self.get_eos()

        # create substrate mask
        substrate_position_mask = torch.zeros_like(logits)
        # mask all other tokens at first position
        substrate_position_mask[:, 0] = 1
        # mask substrate at all other positions
        substrate_index_mask = torch.zeros_like(logits)
        substrate_index_mask[:, :, substrate] = 1
        substrate_mask = torch.logical_xor(substrate_position_mask, substrate_index_mask)

        # get index of last material to bound air block
        not_air = logits.argmax(dim = -1, keepdim = True).ne(air) # (batch, |coating|, |materials_embedding|)
        # logical or along dimension -1
        not_air = not_air.int().sum(dim = -1).bool() # (batch, |coating|)
        not_air_rev = not_air.flip(dims=[1]).to(torch.int) # (batch, |coating|)
        last_mat_idx_rev = torch.argmax(not_air_rev, dim = -1) # (batch)
        last_mat_idx = not_air.shape[-1] - last_mat_idx_rev - 1  # (batch)
        # ensure at least last element is air
        last_mat_idx = torch.minimum(last_mat_idx, torch.tensor(not_air.shape[-1] - 2))
        # create air mask
        range_coating = torch.arange(seq_len, device=CM().get('device')) # (|coating|)
        # mask all other tokens at air block
        air_position_mask = range_coating[None] >= (last_mat_idx + 1)[:, None] # (batch, |coating|)
        air_position_mask = air_position_mask[:, :, None].repeat(1, 1,
                                                                 self.tgt_vocab_size) # (batch, |coating|, |vocab|)
        # mask air at all other positions
        air_index_mask = torch.zeros_like(logits)
        air_index_mask[:, :, air] = 1
        air_mask = torch.logical_xor(air_position_mask, air_index_mask)

        # mask logits
        mask = torch.logical_or(substrate_mask, air_mask)
        subtrahend = torch.zeros_like(logits)
        subtrahend[mask] = torch.inf
        logits = logits - subtrahend
        # logits.masked_fill_(mask == 1, -torch.inf)
        return logits

    def print_coatings_in_parallel(self, coating_1: Coating, coating_2: Coating):
        string_1 = str(coating_1)
        string_2 = str(coating_2)
        max_length = max(len(line) for line in string_1.split('\n'))
        for line_1, line_2 in zip(string_1.split('\n'), string_2.split('\n')):
            print(f"{line_1.ljust(max_length)} | {line_2}")

    def reset_normalisation_values(self):
        self.handle_new_data(self.dataloader.dataset)

    def evaluate_epoch(self, epoch: int):
        # visualise first item of batch
        self.model.eval()
        reflectivity, coating_encoding = self.dataloader[:1]
        reflectivity = reflectivity.to(CM().get('device'))
        if self.density == 'explicit':
            coating_encoding = None
        else:
            coating_encoding = coating_encoding.to(CM().get('device'))
        reflectivity = torch.stack([reflectivity[:, :self.src_seq_len], reflectivity[:, self.src_seq_len:]], dim = 2)
        refs = ReflectivityPattern(reflectivity[:, :, 0], reflectivity[:, :, 1])
        with torch.no_grad():
            thicknesses, unmasked_logits = self.model_call(reflectivity, coating_encoding)

            # masked_logits = self.mask_logits(unmasked_logits)
            unmasked_materials = self.logits_to_indices(unmasked_logits)
            unmasked_output = torch.cat([thicknesses, unmasked_materials], dim = -1)
            # masked_materials = self.logits_to_indices(masked_logits)
            # masked_output = torch.cat([thicknesses, masked_materials], dim = -1)

        unmasked_coating = Coating(unmasked_output)
        # masked_coating = Coating(masked_output)
        preds = coating_to_reflectivity(unmasked_coating) # this used to be masked coating
        if coating_encoding is not None:
            self.print_coatings_in_parallel(unmasked_coating.get_batch(0), Coating(coating_encoding).get_batch(0))
        else:
            print(unmasked_coating.get_batch(0))
        if CM().get('wandb.log') or CM().get('wandb.sweep'):
            validation_error = sum(evaluate_model(self).values())
            self.reset_normalisation_values()
            wandb.log({"validation error": validation_error})
            structure_error = self.compute_structure_error(unmasked_coating)
            wandb.log({"structure error": structure_error})
        visualise(preds, refs, f"{self.get_architecture_name()}/from_training_epoch_{epoch}")

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
        props_dict = {
            "architecture": self.get_architecture_name(),
            "model_details": CM().get(self.get_architecture_name()),
            "num_layers": CM().get('num_layers'),
            "min_wl": CM().get('wavelengths')[0].item(),
            "max_wl": CM().get('wavelengths')[-1].item(),
            "wl_step": len(CM().get('wavelengths')),
            "polarisation": CM().get('polarisation'),
            "materials_hash": EM().hash_materials(),
            "num_materials": len(CM().get('materials.thin_films')),
            "theta": CM().get('theta').item(),
            "air_pad": CM().get('air_pad'),
            "stratified_sampling": CM().get('stratified_sampling'),
            "tolerance": CM().get('tolerance'),
            "num_points": CM().get('training.dataset_size'),
            "epochs": self.num_epochs,
            "curriculum": CM().get('training.curriculum'),
        }
        model_filename = get_unique_filename(f"out/models/model_{short_hash(props_dict)}.pt")

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
        if CM().get('wandb.log') or CM().get('wandb.sweep'):
            wandb.log({"saved under": model_filename})

    def initialise_weights(self, model: nn.Module):
        """Initialise model linear weights using Kaiming normal initialisation."""
        # TODO: add itialisation for other layer types
        if isinstance(model, nn.Linear):
            init.kaiming_normal_(model.weight, nonlinearity = 'relu')
            if model.bias is not None:
                init.zeros_(model.bias)

    def init_dataloader(self):
        """Initialise dataloader for first leg of curriculum."""
        self.dataloader = DynamicDataloader(batch_size=CM().get('training.batch_size'), shuffle=True)
        try:
            self.dataloader.load_leg(0)
        except FileNotFoundError:
            raise FileNotFoundError("Dataset in current configuration not found. Please run generate_dataset.py first.")

    def model_predict(self, target: ReflectivityPattern):
        """
        Predict a coating given a reflectivity pattern object.

        Args:
            target: Reflectivity pattern for which to perform prediction.
        """
        self.model.eval()
        # convert input to shape expected by model
        model_input = torch.stack((target.get_lower_bound(), target.get_upper_bound()), dim = 2)
        # predict encoding of coating
        thicknesses, logits = self.model_call(model_input)
        # logits = self.mask_logits(logits)
        materials = self.logits_to_indices(logits)
        preds = torch.cat([thicknesses, materials], dim = -1)
        return Coating(preds)

    def model_predict_raw(self, target):
        self.model.eval()
        model_input = torch.stack((target.get_lower_bound(), target.get_upper_bound()), dim = 2)
        thicknesses, logits = self.model_call(model_input)
        return thicknesses, F.softmax(logits, dim = -1)

    def compute_memorisation_score(self, input: Coating, labels: Coating):
        input_materials = input.get_material_indices()
        label_materials = labels.get_material_indices()
        num_points = input_materials.shape[0] * input_materials.shape[1]
        accuracy = input_materials.eq(label_materials).sum() / num_points

        input_thicknesses = input.get_thicknesses()
        label_thicknesses = labels.get_thicknesses()
        r2 = 1 - torch.sum((label_thicknesses - input_thicknesses)**2) / torch.sum((label_thicknesses - label_thicknesses.mean())**2)

        return accuracy, r2

    def compute_loss_guided(self, input: tuple[torch.Tensor], labels: torch.Tensor):
        """
        Compute guided loss of a batch of data.

        Guided loss is the scaled L2 loss between the predicted coating and the ground truth coating.

        Args:
            input: Model output.
            labels: Ground truth coating encoding.
        """
        input_thicknesses, input_logits = input
        batch_size, seq_len, _ = input_thicknesses.shape
        # TODO: make the shapes pretty here
        label_thicknesses = labels[:, :, :1]
        label_materials = labels[:, :, 1]

        # ignoring substrate index is actually very important here because the sequential models prepend their logits with [1, -inf, -inf, ...] which leads to inf cross-entropy for that position when label_smoothing is used
        material_loss = F.cross_entropy(input_logits.transpose(1, 2), label_materials.to(torch.long), label_smoothing = 0.1, reduction = 'none', ignore_index = EM().get_substrate_index()) / batch_size
        material_loss = material_loss.mean() / (material_loss.norm(2) + 1.e-9)
        thickness_loss = F.mse_loss(input_thicknesses, label_thicknesses, reduction = 'none') / batch_size
        thickness_loss = thickness_loss.mean() / (thickness_loss.norm(2) + 1.e-9)
        if CM().get('wandb.log') or CM().get('wandb.sweep'):
            wandb.log({
                'material_loss': material_loss.item(),
                'thickness_loss': thickness_loss.item(),
            })
        return thickness_loss, material_loss

    def compute_loss_free(self, input: tuple[torch.Tensor], labels: torch.Tensor):
        """
        Compute free loss of a batch of data.

        Free loss is the L2 loss between the predicted reflectivity value and the ground truth reflective properties pattern.

        Args:
            batch: Tuple of features and labels for which to compute loss.
        """
        # create reflectivity pattern for match operation
        pattern = ReflectivityPattern(labels[:, :, 0], labels[:, :, 1])
        input_thicknesses, input_logits = input
        input_materials = self.logits_to_indices(input_logits)
        coating_encoding = torch.cat([input_thicknesses, input_materials], dim = -1)
        coating = Coating(coating_encoding)
        # convert predicted coating to reflectivity value
        preds = coating_to_reflectivity(coating)
        free_loss = match(preds, pattern, reduction = 'none')
        free_loss = free_loss.mean() / (free_loss.norm(2) + 1.e-9)
        constraint_loss = self.compute_constraint_loss(coating, reduction = 'none')
        constraint_loss = constraint_loss.mean() / (constraint_loss.norm(2) + 1.e-9)
        if CM().get('wandb.log') or CM().get('wandb.sweep'):
            wandb.log({
                'free_only_loss': free_loss.item(),
                'constraint_loss': constraint_loss.item(),
            })
        return free_loss

    def compute_constraint_loss(self, coating: Coating, reduction: str = 'mean'):
        thicknesses = coating.get_thicknesses()
        batch_size, seq_len = thicknesses.shape
        loss = (F.relu(thicknesses - CM().get('thicknesses_max')) ** 2) / (batch_size * seq_len)
        if reduction == 'none':
            return loss
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss.mean()

    def compute_structure_error(self, coating: Coating):
        materials = coating.get_encoding()[:, :, 1:]
        thicknesses = coating.get_encoding()[:, :, :1]
        batch_size, seq_len, _ = materials.shape
        scale_factor = batch_size * seq_len

        substrate = self.get_bos().to(torch.float)
        air = self.get_eos().to(torch.float)

        # make mask for substrate
        substrate_mask = torch.zeros_like(materials).to(CM().get('device')).to(
            torch.int) # (batch, |coating|, |materials_embedding|)
        substrate_mask[:, 0] = 1
        substrate_mask = substrate_mask.bool()
        # test that substrate is at first position
        substrate_pos = substrate_mask * substrate
        # test that substrate is nowhere else
        substrate_neg = ~substrate_mask * substrate

        # get index of last material before air
        not_air = materials.ne(air) # (batch, |coating|, |materials_embedding|)
        # logical or along dimension -1
        not_air = not_air.int().sum(dim = -1).bool() # (batch, |coating|)
        not_air_rev = not_air.flip(dims=[1]).to(torch.int) # (batch, |coating|)
        last_mat_idx_rev = torch.argmax(not_air_rev, dim = -1) # (batch)
        last_mat_idx = not_air.shape[-1] - last_mat_idx_rev - 1  # (batch)
        # ensure at least last element is air
        last_mat_idx = torch.minimum(last_mat_idx, torch.tensor(not_air.shape[-1] - 2))
        # make mask for air block
        range_coating = torch.arange(seq_len, device=CM().get('device')) # (|coating|)
        air_mask = range_coating[None] >= (last_mat_idx + 1)[:, None] # (batch, |coating|)
        air_mask = air_mask[:, :, None] # (batch, |coating|, 1)
        # test that air block is at the end
        air_pos = torch.zeros_like(materials).to(CM().get('device')) # (batch, |coating|, |materials_embedding|)
        air_pos[air_mask] = air.squeeze().repeat(air_pos[air_mask].shape[0])
        # test that air is nowhere else
        air_neg = torch.zeros_like(materials).to(CM().get('device')) # (batch, |coating|, |materials_embedding|)
        air_neg[~air_mask] = air.squeeze().repeat(air_neg[~air_mask].shape[0])

        # calculate distances
        substrate_pos_err = torch.sum(materials.ne(substrate_pos) * substrate_mask)
        substrate_neg_err = torch.sum(materials.eq(substrate_neg) * ~substrate_mask)
        air_pos_err = torch.sum(materials.ne(air_pos) * air_mask)
        air_neg_err = torch.sum(materials.eq(air_neg) * ~air_mask)

        materials_err = (substrate_pos_err + substrate_neg_err + air_pos_err + air_neg_err) / scale_factor

        thicknesses_err = (thicknesses.numel() - thicknesses.count_nonzero()) / (batch_size * seq_len)

        return materials_err + thicknesses_err

    def model_call(self, src, tgt = None):
        src = self.normalise_reflectivity(src)
        if tgt is not None:
            input_thicknesses, input_materials = tgt.chunk(2, -1)
            input_thicknesses = self.normalise_thicknesses(input_thicknesses)
            input_materials = self.normalise_materials(input_materials)
            tgt = torch.cat([input_thicknesses, input_materials], dim = -1)

        output_thicknesses, output_material_logits = self.get_model_output(src, tgt)

        output_thicknesses = torch.clamp(self.denormalise_thicknesses(output_thicknesses), 0, 10_000)
        return output_thicknesses, output_material_logits

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

    def load_or_train(self, attributes: dict = None):
        architecture = self.get_architecture_name()
        if attributes is None:
            attributes = {'architecture': architecture}
        else:
            attributes = {**{'architecture': architecture}, **attributes}
        model_filename = get_saved_model_path(attributes=attributes)
        if model_filename is None:
            print(f"Saved {architecture} model not found. Performing training...")
            # update config attributes
            CM().set(attributes)
            if CM().get('wandb.log'):
                wandb.init(
                    project=CM().get('wandb.project'),
                    config=CM().get('wandb.config')
                )
            self.train()
            # restore config to original state
            CM().reset()
        else:
            self.model = torch.load(model_filename, weights_only=False, map_location = CM().get('device'))

    def scale_gradients(self):
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                norm = param.grad.data.norm(2)
                param.grad.data = param.grad.data / (norm + 1.0e-10)

    def get_num_params(self):
        return sum(p.numel() for p in self.model.parameters())