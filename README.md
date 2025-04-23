# Inverse Multilayer Optics
## Overview
This project aims to develop a neural network-based approach to inverse the reflectivity function of a thin film coating. Given a reflectivity pattern, the goal is to predict the optimal coating configuration.

## Usage

### Install submodule
Run `git submodule update --init tmm_clean` to add tmm submodule.
### Configure
Specify the details of your experiment in `config.yaml`.
### Generate dataset
Run `generate_dataset.py` to create training data or `generate_dataset.py validation` to create validation data.
### Train a model
Run `train_model.py` to train a new model.
### Score a model
Run `score_model.py <model_name>` to load and evaluate a saved model on validation and test data.
### Example Use Cases
Two pre-trained models are provided:

`model_free_switch_1200.pt`
`model_guided_switch_1200.pt`<br>
You can score these models using<br>`score_model.py model_free_switch_1200.pt` and<br>`score_model.py model_guided_switch_1200.pt`.

## Requirements
Python 3.10.12<br>
PyTorch 2.1.1<br>
NumPy 1.26.2<br>
Matplotlib 3.10.0<br>
SciPy 1.15.1<br>
WandB 0.19.7
