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

            # set materials embedding dim
            substrate_title = self.config['materials']['substrate']
            with open(self.config['material_embedding']['data_file'], 'r') as file:
                material_data = yaml.safe_load(file)["materials"]
            # filter out materials with wrong name
            material_data = list(filter(lambda d: d["title"] == substrate_title, material_data))
            # count defining dimensions per material
            material_data = list(map(lambda d: {"title": d["title"], "dim": sum([len(d[key]) if isinstance(d[key], list) else 1 for key in [k for k in d.keys() if k != "title"]])}, material_data))
            assert len(material_data) == 1, "More than one substrate provided"
            defining_dim = material_data[0]["dim"]
            specified_dim = self.config['material_embedding']['dim']
            self.config['material_embedding']['dim'] = min(specified_dim, defining_dim)

            self.config['training']['num_legs'] = len(self.config['training']['guidance_schedule'])

        except BaseException as e:
            raise ValueError(f"Error loading config: {e}")
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

    def set_layers_to(self, num_layers: int):
        self.config["layers"]["max"] = num_layers
        self.config["num_layers"] = num_layers

    def set_seed_to(self, seed: int):
        random.seed(seed)
        torch.manual_seed(seed)