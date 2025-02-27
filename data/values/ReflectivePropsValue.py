import torch
from data.values.BaseReflectiveProps import BaseReflectiveProps

class ReflectivePropsValue(BaseReflectiveProps):
    def __init__(self, value: torch.Tensor):
        assert len(value.shape) == 2
        super().__init__()
        self.value = value

    def get_value(self):
        return self.value

    def to(self, device: str):
        return ReflectivePropsValue(self.value.to(device))

    def device(self):
        return self.value.device

    def __eq__(self, other):
        if isisntance(other, ReflectivePropsValue):
            return torch.equal(self.value, other.get_value())
        return False

    def __str__(self):
        return f"Reflective Props Value object:\n\tvalue: {self.value.squeeze().cpu().detach().numpy()}"