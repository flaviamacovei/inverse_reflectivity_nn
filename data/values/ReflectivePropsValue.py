import torch
from data.values.BaseReflectiveProps import BaseReflectiveProps

class ReflectivePropsValue(BaseReflectiveProps):
    def __init__(self, start_wl: int, end_wl: int, value: torch.Tensor):
        assert len(value.shape) == 1
        super().__init__(start_wl, end_wl, value.shape[0])
        self.value = value

    def get_value(self):
        return self.value