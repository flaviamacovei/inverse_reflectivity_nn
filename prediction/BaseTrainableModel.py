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

        # shape variables
        self.src_seq_len = CM().get('wavelengths').shape[0]
        self.src_dim = 2 # lower bound and upper bound
        self.tgt_seq_len = CM().get('num_layers') + 2 # thin films + substrate + air
        self.tgt_vocab_size = len(CM().get('materials.thin_films')) + 2 # available thin films + substrate + air
        self.tgt_dim = CM().get('material_embedding.dim')
        self.in_dims = {'seq_len': self.src_seq_len, 'dim': self.src_dim}
        self.out_dims = {'seq_len': self.tgt_seq_len, 'material': self.tgt_vocab_size, 'thickness': 1}

        self.init_dataloader()

        # loss function depends on guidance of current leg
        self.loss_functions = {
            "free": self.compute_loss_free,
            "guided": self.compute_loss_guided
        }
        self.compute_loss = None
        self.current_leg = -1
        self.guidance = None
        self.checkpoint = None

        sampling_functions = {
            "soft": self.soft_sample,
            "greedy": self.greedy_sample
        }
        self.sample = sampling_functions[CM().get('sampling')]

        self.vocab = EM().get_embeddings()

        self.model = self.build_model()
        self.optimiser = torch.optim.Adam(self.model.parameters())
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimiser, mode = 'min', patience = 5, cooldown = 5)

    @abstractmethod
    def build_model(self):
        """
        Returns a model that extends nn.Module based on the architecture of the subclass.
        """
        pass

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
            # update attributes: loss function, learning rate, dataloader,
            # first run data through model, then compute loss
            # shapes might be off (+ 1 due to changes in SegmentedDataset)
            self.compute_loss = self.loss_functions[guidance]
            for g in self.optimiser.param_groups:
                g['lr'] = CM().get(f'training.{guidance}_learning_rate')
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
            for i, batch in enumerate(self.dataloader):
                self.optimiser.zero_grad()
                reflectivity, coating_encoding = batch
                reflectivity = reflectivity.to(CM().get('device'))
                reflectivity = torch.stack([reflectivity[:, :self.src_seq_len], reflectivity[:, self.src_seq_len:]], dim = 2)

                coating_encoding = coating_encoding.to(CM().get('device'))
                # a batch consists of reflectivity and coating encodings
                # in free mode, reflectivity is simultaneously input and ground truth
                # in guided mode, coating encodings are ground truth
                labels = reflectivity if self.guidance == 'free' else coating_encoding

                output = self.get_model_output(reflectivity, coating_encoding)

                loss = self.compute_loss(output, labels)
                # print(loss.item())
                epoch_loss += loss.item()

                if CM().get('wandb.log'):
                    wandb.log({"loss": loss.item()})
                    # wandb.log({"lr": self.optimiser.param_groups[0]['lr']})

                loss.backward()

                # for name, param in self.model.named_parameters():
                #     if param.requires_grad:
                #         print(name, param.data)

                self.scale_gradients()
                self.optimiser.step()
            epoch_loss /= len(self.dataloader)
            # self.scheduler.step(epoch_loss)
            if epoch % max(1, CM().get('training.num_epochs') / 20) == 0:
                if CM().get('training.save_model'):
                    # logging at every 5% of training
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

    def get_bos(self):
        substrate = EM().get_material(CM().get('materials.substrate'))
        bos = EM().encode([substrate])[None] # (batch, |coating| = 1, 1)
        return bos

    def get_eos(self):
        air = EM().get_material(CM().get('materials.air'))
        eos = EM().encode([air])[None] # (batch, |coating| = 1, 1)
        return eos

    def get_bos_logits(self):
        bos = self.get_bos()
        bos_index = self.vocab.eq(bos).nonzero(as_tuple=True)[0].item()
        bos_logits = torch.ones(1, 1, self.tgt_vocab_size).to(CM().get('device'))
        bos_logits = bos_logits * -torch.inf
        bos_logits[:, :, bos_index] = 1
        return bos_logits

    def soft_sample(self, logits: torch.Tensor):
        softmax_probabilities = F.softmax(logits, dim = -1)
        return softmax_probabilities @ self.vocab

    def greedy_sample(self, logits: torch.Tensor):
        # compute soft probabilities of nearest neighbours
        # this vector remains connected to the computational graph to maintain differentiability
        greedy_probs_soft = torch.exp(-torch.abs(logits))
        # compute hard probabilities of nearest neighbours
        # this is a one-hot vector but disconnected from the computational graph
        greedy_probs_hard = torch.zeros_like(greedy_probs_soft).scatter_(-1, greedy_probs_soft.argmin(dim=-1, keepdim=True), 1.0)
        # attach computational graph from soft probabilities to hard probabilities
        greedy_probabilities = greedy_probs_hard + (greedy_probs_soft - greedy_probs_soft.detach())
        return greedy_probabilities @ self.vocab

    def mask_logits(self, logits: torch.Tensor):
        return logits
        seq_len = logits.shape[1]
        # logits = logits.reshape(logits.shape[0], self.seq_len, self.vocab_size)
        substrate = EM().get_material(CM().get('materials.substrate'))
        air = EM().get_material(CM().get('materials.air'))
        substrate_encoding, air_encoding = EM().encode([substrate, air])
        # get index of substrate and air in embeddings lookup
        substrate_index = EM().get_embeddings().eq(substrate_encoding).nonzero(as_tuple=True)[0].item()
        air_index = EM().get_embeddings().eq(air_encoding).nonzero(as_tuple=True)[0].item()

        # create substrate mask
        substrate_position_mask = torch.zeros_like(logits)
        # mask all other tokens at first position
        substrate_position_mask[:, 0] = 1
        # mask substrate at all other positions
        substrate_index_mask = torch.zeros_like(logits)
        substrate_index_mask[:, :, substrate_index] = 1
        substrate_mask = torch.logical_xor(substrate_position_mask, substrate_index_mask)

        # get index of last material to bound air block
        not_air = logits.argmax(dim=-1, keepdim=True).ne(air_index)  # (batch, |coating|, |materials_embedding|)
        # logical or along dimension -1
        not_air = not_air.int().sum(dim=-1).bool()  # (batch, |coating|)
        not_air_rev = not_air.flip(dims=[1]).to(torch.int)  # (batch, |coating|)
        last_mat_idx_rev = torch.argmax(not_air_rev, dim=-1)  # (batch)
        last_mat_idx = not_air.shape[-1] - last_mat_idx_rev - 1  # (batch)
        # ensure at least last element is air
        last_mat_idx = torch.minimum(last_mat_idx, torch.tensor(not_air.shape[-1] - 2))
        # create air mask
        range_coating = torch.arange(seq_len, device=CM().get('device'))  # (|coating|)
        # mask all other tokens at air block
        air_position_mask = range_coating[None] >= (last_mat_idx + 1)[:, None]  # (batch, |coating|)
        air_position_mask = air_position_mask[:, :, None].repeat(1, 1, self.vocab_size)  # (batch, |coating|, |vocab|)
        # mask air at all other positions
        air_index_mask = torch.zeros_like(logits)
        air_index_mask[:, :, air_index] = 1
        air_mask = torch.logical_xor(air_position_mask, air_index_mask)

        # mask logits
        mask = torch.logical_or(substrate_mask, air_mask)
        subtrahend = torch.zeros_like(logits)
        subtrahend[mask] = torch.inf
        logits = logits - subtrahend
        # logits.masked_fill_(mask == 1, -torch.inf)
        return logits

    def visualise_epoch(self, epoch: int):
        # visualise first item of batch
        self.model.eval()
        first_batch = self.dataloader[0]
        reflectivity, coating_encoding = first_batch
        reflectivity = reflectivity.to(CM().get('device'))
        coating_encoding = coating_encoding.to(CM().get('device'))
        reflectivity = reflectivity[None]
        reflectivity = torch.stack([reflectivity[:, :self.src_seq_len], reflectivity[:, self.src_seq_len:]], dim=2)
        coating_encoding = coating_encoding[None]
        refs = ReflectivityPattern(reflectivity[:, :, 0], reflectivity[:, :, 1])
        with torch.no_grad():
            output = self.get_model_output(reflectivity, coating_encoding)

            thicknesses = output[:, :, :1]
            logits = output[:, :, 1:]
            materials = self.sample(logits)
            output = torch.cat([thicknesses, materials], dim = -1)

        coating = Coating(output)
        preds = coating_to_reflectivity(coating)
        print(coating.get_batch(0))
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
            "epochs": CM().get('training.num_epochs'),
            "guidance_schedule": CM().get('training.guidance_schedule'),
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

    def predict(self, target: ReflectivityPattern):
        """
        Predict a coating given a reflectivity pattern object.

        Args:
            target: Reflectivity pattern for which to perform prediction.
        """
        self.model.eval()
        # convert input to shape expected by model
        model_input = torch.stack((target.get_lower_bound(), target.get_upper_bound()), dim = 2)
        # predict encoding of coating
        output = self.get_model_output(model_input)
        thicknesses = output[:, :, :1]
        logits = output[:, :, 1:]
        # logits = self.mask_logits(logits)
        materials = self.sample(logits)
        preds = torch.cat([thicknesses, materials], dim=-1)
        return Coating(preds)

    def get_coating_indices(self, coating: torch.Tensor):
        coating = coating[:, :, None].repeat(1, 1, self.tgt_vocab_size, 1) # (batch, seq_len, vocab_size, embed_dim)
        coating_eq = coating.eq(self.vocab) # (batch, seq_len, vocab_size, embed_dim)
        coating_indices = coating_eq.prod(dim = -1) # (batch, seq_len, vocab_size)
        return coating_indices.argmax(dim = -1) # (batch, seq_len)

    def compute_loss_guided(self, input: torch.Tensor, labels: torch.Tensor):
        """
        Compute guided loss of a batch of data.

        Guided loss is the scaled L2 loss between the predicted coating and the ground truth coating.

        Args:
            input: Model output.
            labels: Ground truth coating encoding.
        """
        batch_size, seq_len, _ = input.shape
        scale_mean = batch_size * seq_len
        input_thicknesses = input[:, :, :1]
        input_logits = input[:, :, 1:]
        label_thicknesses = labels[:, :, :1]
        label_materials = labels[:, :, 1:]

        indices = self.get_coating_indices(label_materials)
        softmax_probabilities = F.softmax(input_logits, dim = -1)
        material_loss = F.cross_entropy(softmax_probabilities.transpose(1, 2), indices) / batch_size
        # material_loss = F.mse_loss(self.sample(input_logits), label_materials)
        # material_loss = 0
        thickness_loss = F.mse_loss(input_thicknesses, label_thicknesses) / batch_size
        # thickness_loss = 0
        loss = material_loss + thickness_loss

        regularisation_loss = self.regularise(Coating(torch.cat([input_thicknesses, self.sample(input_logits)], dim = -1)))
        lmd = CM().get('training.reg_weight')
        return loss + lmd * regularisation_loss

    def compute_loss_free(self, input: torch.Tensor, labels: torch.Tensor):
        """
        Compute free loss of a batch of data.

        Free loss is the L2 loss between the predicted reflectivity value and the ground truth reflective properties pattern.

        Args:
            batch: Tuple of features and labels for which to compute loss.
        """
        # create reflectivity pattern for match operation
        pattern = ReflectivityPattern(labels[:, :, 0], labels[:, :, 1])
        input_thicknesses = input[:, :, :1]
        input_logits = input[:, :, 1:]
        input_materials = self.sample(input_logits)
        coating_encoding = torch.cat([input_thicknesses, input_materials], dim = -1)
        coating = Coating(coating_encoding)
        # convert predicted coating to reflectivity value
        preds = coating_to_reflectivity(coating)
        loss = match(preds, pattern)
        regularisation = self.regularise(coating)
        lmd = CM().get('training.reg_weight')
        return loss + lmd * regularisation

    def regularise(self, coating: Coating):
        return 0
        materials = coating.get_encoding()[:, :, 1:]
        thicknesses = coating.get_encoding()[:, :, :1]
        batch_size, seq_len, encoding_size = materials.shape
        scale_factor = batch_size * seq_len * encoding_size

        substrate = self.get_bos().reshape(encoding_size)
        air = self.get_eos().reshape(encoding_size)

        # make mask for substrate
        substrate_mask = torch.zeros_like(materials).to(CM().get('device')).to(torch.int) # (batch, |coating|, |materials_embedding|)
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
        not_air_rev = not_air.flip(dims = [1]).to(torch.int) # (batch, |coating|)
        last_mat_idx_rev = torch.argmax(not_air_rev, dim = -1) # (batch)
        last_mat_idx = not_air.shape[-1] - last_mat_idx_rev - 1 # (batch)
        # ensure at least last element is air
        last_mat_idx = torch.minimum(last_mat_idx, torch.tensor(not_air.shape[-1] - 2))
        # make mask for air block
        range_coating = torch.arange(seq_len, device=CM().get('device')) # (|coating|)
        air_mask = range_coating[None] >= (last_mat_idx + 1)[:, None]  # (batch, |coating|)
        air_mask = air_mask[:, :, None].expand(-1, -1, encoding_size) # (batch, |coating|, |materials_embedding|)
        # test that air block is at the end
        air_pos = torch.zeros_like(materials).to(CM().get('device')) # (batch, |coating|, |materials_embedding|)
        air_pos[air_mask] = air.repeat(air_pos[air_mask].shape[0] // encoding_size)
        # test that air is nowhere else
        air_neg = torch.zeros_like(materials).to(CM().get('device')) # (batch, |coating|, |materials_embedding|)
        air_neg[~air_mask] = air.repeat(air_neg[~air_mask].shape[0] // encoding_size)

        # calculate distances
        substrate_pos_err = torch.sum(materials.ne(substrate_pos) * substrate_mask)
        substrate_neg_err = torch.sum(materials.eq(substrate_neg) * ~substrate_mask)
        air_pos_err = torch.sum(materials.ne(air_pos) * air_mask)
        air_neg_err = torch.sum(materials.eq(air_neg) * ~air_mask)

        materials_err = (substrate_pos_err + substrate_neg_err + air_pos_err + air_neg_err) / scale_factor

        thicknesses_err = (thicknesses.numel() - thicknesses.count_nonzero()) / (batch_size * seq_len)
        # thicknesses_err = 0

        return materials_err + thicknesses_err

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
        model_filename = get_saved_model_path(attributes = attributes)
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
            self.model = torch.load(model_filename, weights_only = False)
            self.model = self.model.to(CM().get('device'))


    @abstractmethod
    def scale_gradients(self):
        """Scale gradients of trainable model. Must be implemented by subclasses."""
        ...

    def get_num_params(self):
        return sum(p.numel() for p in self.model.parameters())