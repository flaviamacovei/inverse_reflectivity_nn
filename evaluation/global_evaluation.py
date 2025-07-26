import pandas as pd
import sys
sys.path.append(sys.path[0] + '/..')
from utils.data_utils import load_config
from utils.ConfigManager import ConfigManager as CM
from data.dataloaders.DynamicDataloader import DynamicDataloader

def random_baseline():
    def load_config(config_id: int):
        dataloader = DynamicDataloader(CM().get('training.dataset_size'), shuffle = False)
        filepath = get_dataset_name("training", density)