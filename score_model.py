import sys
sys.path.append(sys.path[0] + '/..')
from prediction.GradientModel import GradientModel
from prediction.MLP import MLP
from prediction.CNN import CNN
from prediction.Transformer import Transformer
from utils.ConfigManager import ConfigManager as CM
from evaluation.model_eval import evaluate_model, test_model

def score_model(model):
    """Evaluate model on validation and test data."""
    evaluate_model(model)
    test_model(model)

if __name__ == "__main__":
    # model name must be passed as first argument
    saved_model_file = sys.argv[1]
    # map architecture specified in config to model classes
    models = {
        "gradient": GradientModel,
        "mlp": MLP,
        "cnn": CNN,
        "transformer": Transformer
    }
    Model = models[CM().get('architecture')]

    model = Model()
    model.load(f"out/models/{saved_model_file}")
    score_model(model)
