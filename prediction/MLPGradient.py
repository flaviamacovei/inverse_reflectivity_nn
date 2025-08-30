import torch
import sys
sys.path.append(sys.path[0] + '/..')
from prediction.BaseTrainableModel import BaseTrainableModel
from prediction.MLP import MLP
from prediction.BaseModel import BaseModel
from prediction.GradientModel import GradientModel
from data.dataloaders.BaseDataloader import BaseDataloader
from utils.ConfigManager import ConfigManager as CM
from data.values.ReflectivityPattern import ReflectivityPattern

class MLPGradient(MLP):
    """
    Trainable prediction model using an MLP as base.

    Attributes:
        model: Instance of TrainableMLP.
    """
    def __init__(self):
        """Initialise an MLPGradient instance."""
        super().__init__()
        self.gradient = GradientModel()

    def predict(self, target: ReflectivityPattern):
        mlp_prediction = super().predict(target)
        print(mlp_prediction.get_encoding().shape)
        self.gradient.initialise(mlp_prediction.get_encoding().detach().cpu().numpy().flatten())
        gradient_prediction = self.gradient.predict(target)
        return gradient_prediction
