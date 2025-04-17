import random
import torch
import yaml
import numpy as np

import os
dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
config_path = os.path.join(dir, 'config.yaml')

class ConfigManager:
    """
    Configuration manager for loading and accessing configuration settings.

    This class is a singleton.

    Attributes:
        config: Dictionary containing the loaded configuration settings.

    Methods:
        get: Return value of configuration setting by specified key.
    """
    _instance = None

    def __new__(cls, dir = dir):
        """Initialise a ConfigManager instance."""
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._load(dir)
        return cls._instance

    def _load(self, dir):
        """Load configuration settings and perform transformations where necessary."""
        with open(dir + '/config.yaml', 'r') as f:
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

            # set materials location
            self.config['material_embedding']['data_file'] = os.path.join(dir, self.config['material_embedding']['data_file'])

            self.config['training']['num_legs'] = len(self.config['training']['guidance_schedule'])

        except BaseException as e:
            print(f"Error loading config: {e}")
        return self.config

    def get(self, key, default=None):
        """Return value of configuration setting by specified key."""
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                # current level is dictionary, access by key
                value = value[k]
            elif isinstance(value, list) and k.isdigit():
                # current level is list, access by index
                index = int(k)
                if index < len(value):
                    value = value[index]
                else:
                    return default
            else:
                return default
        return value