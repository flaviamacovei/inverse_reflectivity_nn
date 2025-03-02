import random
import torch
import yaml
import numpy as np

class ConfigManager:
    _instance = None

    def __new__(cls, filepath="config.yaml"):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._load(filepath)
        return cls._instance

    def _load(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        try:
            if self.config['device'] == 'auto':
                self.config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
            seed = self.config['seed']
            if seed:
                random.seed(seed)
                torch.manual_seed(seed)
            self.config['wavelengths'] = torch.linspace(self.config['wavelengths']['start'], self.config['wavelengths']['end'], self.config['wavelengths']['steps'], device = self.config['device'])
            self.config['theta'] = torch.tensor(np.linspace(self.config['theta']['start'], self.config['theta']['end'], self.config['theta']['steps']) * (np.pi / 180), dtype = torch.float32).to(self.config['device'])
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