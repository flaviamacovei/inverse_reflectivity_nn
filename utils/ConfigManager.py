import random
import torch
import yaml
import numpy as np

import os
config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'config.yaml')

class ConfigManager:
    _instance = None

    def __new__(cls, filepath=config_path):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._load(filepath)
        return cls._instance

    def _load(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        try:
            # set device
            if self.config['device'] == 'auto':
                self.config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

            # set random seed
            seed = self.config['seed']
            if seed:
                random.seed(seed)
                torch.manual_seed(seed)

            # make linspaces
            self.config['wavelengths'] = torch.linspace(self.config['wavelengths']['start'], self.config['wavelengths']['end'], self.config['wavelengths']['steps'], device = self.config['device'])
            self.config['theta'] = torch.tensor(np.linspace(self.config['theta']['start'], self.config['theta']['end'], self.config['theta']['steps']) * (np.pi / 180), dtype = torch.float32).to(self.config['device'])

            # set dataset
            if self.config['training']['guidance'] == "free":
                self.config['dataset_files'] = [f"free_complete_{self.config['training']['dataset_size']}.pt", f"free_masked_{self.config['training']['dataset_size']}.pt", f"free_explicit_{self.config['training']['dataset_size']}.pt"]
            elif self.config['training']['guidance'] == "guided":
                self.config['dataset_files'] = [f"guided_complete_{self.config['training']['dataset_size']}.pt", f"guided_masked_{self.config['training']['dataset_size']}.pt"]
            else:
                raise ValueError (f"Unknown loss function: {self.config['training']['loss_function']}")

        except BaseException as e:
            print(f"Error loading config: {e}")
        return self.config

    def get(self, key, default=None):
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value