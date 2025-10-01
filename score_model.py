import sys
import wandb
sys.path.append(sys.path[0] + '/..')
from prediction.GradientModel import GradientModel
from prediction.MLP import MLP
from prediction.CNN import CNN
from prediction.Transformer import Transformer
from utils.ConfigManager import ConfigManager as CM
from evaluation.model_eval import evaluate_model, test_model

def score_model(model):
    """Evaluate model on validation and test data."""
    log = CM().get('wandb.log') or CM().get('wandb.sweep')
    print("Evaluating model...")
    errors = evaluate_model(model)
    for density in errors.keys():
        print(f"{density}_error: {errors[density]}")
        if log:
            wandb.log({f'{density}_error': errors[density]})
    print(f"total_error: {sum(errors.values())}")
    if log:
        wandb.log({'total_error': sum(errors.values())})
    print("Evaluation complete.")
    print("Testing model...")
    test_error = test_model(model)
    print(f"test error: {test_error}")
    if log:
        wandb.log({"test_error": test_error})
    print("Testing complete.")

if __name__ == "__main__":
    # map architecture specified in config to model classes
    models = {
        "gradient": GradientModel,
        "mlp": MLP,
        "cnn": CNN,
        "transformer": Transformer
    }
    Model = models[CM().get('architecture')]

    model = Model()
    model.load_or_train()
    score_model(model)
