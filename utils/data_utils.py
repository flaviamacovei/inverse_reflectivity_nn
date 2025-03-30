import yaml
import sys
sys.path.append(sys.path[0] + '/..')
from utils.ConfigManager import ConfigManager as CM
from data.material_embedding.EmbeddingManager import EmbeddingManager as EM

def get_dataset_name(partition: str, density: str):
    if partition == "validation":
        guidance = "free"
        num_points = 100
    else:
        guidance = CM().get('training.guidance')
        num_points = CM().get('training.dataset_size')
    props_dict = {
        "partition": partition,
        "num_layers": CM().get('num_layers'),
        "min_wl": CM().get('wavelengths')[0].item(),
        "max_wl": CM().get('wavelengths')[-1].item(),
        "wl_step": len(CM().get('wavelengths')),
        "polarisation": CM().get('polarisation'),
        "materials_hash": EM().hash_materials(),
        "theta": CM().get('theta').item(),
        "tolerance": CM().get('tolerance'),
        "guidance": guidance,
        "density": density,
        "num_points": num_points
    }
    with open("data/datasets/metadata.yaml", "r") as f:
        content = yaml.safe_load(f)
        for dataset in content["datasets"]:
            if dataset["properties"] == props_dict:
                return dataset["title"]
        return None