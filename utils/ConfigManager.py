import random
import torch
import yaml

class ConfigManager:
    _instance = None

    def __new__(cls, filepath="config.yaml"):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._load(filepath)
        return cls._instance

    def _load(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        try:
            if config['device'] == 'auto':
                config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
            seed = config['seed']
            if seed:
                random.seed(seed)
                torch.manual_seed(seed)
            config['wavelengths'] = torch.linspace(config['wavelengths']['start'], config['wavelengths']['end'], config['wavelengths']['steps'], device = config['device'])
            config['theta'] = torch.tensor(np.linspace(config['theta']['start'], config['theta']['end'], config['theta']['steps']) * (np.pi / 180), dtype = torch.float32).to(device)
        except BaseException as e:
            print(f"Error loading config: {e}")
        return config

    def get(self, key, default=None):
        return self.config.get(key, default)