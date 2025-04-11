import sys
from prediction.MLP import MLP
from prediction.CNN import CNN
from prediction.Transformer import Transformer
from utils.ConfigManager import ConfigManager as CM
from evaluation.model_eval import evaluate_model, test_model

if __name__ == "__main__":
    saved_model_file = sys.argv[1]
    models = {
        "mlp": MLP,
        "cnn": CNN,
        "transformer": Transformer
    }
    Model = models[CM().get('architecture')]

    model = Model()
    model.load(f"out/models/{saved_model_file}")

    evaluate_model(model)
    test_model(model)