import torch
from data.values.BaseReflectiveProps import BaseReflectiveProps

class ReflectivePropsValue(BaseReflectiveProps):
    def __init__(self, value: torch.Tensor):
        assert len(value.shape) == 1
        super().__init__()
        self.value = value

    def get_value(self):
        return self.value

    def to(self, device: str):
        return ReflectivePropsValue(self.value.to(device))

    def device(self):
        return self.value.device