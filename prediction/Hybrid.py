import torch
import sys
sys.path.append(sys.path[0] + '/..')
from prediction.BaseTrainableModel import BaseTrainableModel
from prediction.BaseModel import BaseModel
from prediction.GradientModel import GradientModel
from prediction.MLP import MLP
from prediction.CNN import CNN
from prediction.RNN import RNN
from prediction.Transformer import Transformer
from data.dataloaders.BaseDataloader import BaseDataloader
from utils.ConfigManager import ConfigManager as CM

from data.values.ReflectivityPattern import ReflectivityPattern

class Hybrid(BaseTrainableModel):
    """
    Hybrid model using trainable and gradient prediction models.

    Attributes:
        model: Instance of TrainableMLP.
    """
    def __init__(self, trainable_type: str = "transformer"):
        """Initialise Hybrid instance."""
        types = {
            "mlp": MLP,
            "cnn": CNN,
            "rnn": RNN,
            "transformer": Transformer,
        }
        ModelClass = types[trainable_type]
        trainable = ModelClass()
        super().__init__(trainable.model)
        self.trainable = trainable
        self.gradient = GradientModel()

    def get_model_output(self, src, tgt = None):
        return self.trainable.get_model_output(src, tgt)

    def scale_gradients(self):
        self.trainable.scale_gradients()

    def predict(self, target: ReflectivityPattern):
        trainable_prediction = self.trainable.predict(target)
        self.gradient.initialise(trainable_prediction.get_encoding().detach().cpu().numpy().flatten())
        gradient_prediction = self.gradient.predict(target)
        return gradient_prediction

    def get_architecture_name(self):
        """
        Return name of model architecture.
        """
        return f"{self.trainable.get_architecture_name()}+gradient"

    def load_or_train(self):
        self.trainable.load_or_train()
