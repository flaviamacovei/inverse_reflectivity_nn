import torch
from config import tolerance
from forward.forward_tmm import coating_to_reflective_props
from prediction.GradientPredictor import GradientPredictor
from data.values.ReflectivePropsPattern import ReflectivePropsPattern
from data.values.Coating import Coating

if __name__ == "__main__":
    num_layers = 19
    start_wl = 300
    end_wl = 500
    steps = 20

    # make random pattern
    thicknesses = torch.rand((1, num_layers))
    thicknesses[:, 0] = float("Inf")
    thicknesses[:, -1] = float("Inf")
    random_coating = Coating(thicknesses)

    reflective_props_tensor = coating_to_reflective_props(random_coating, start_wl, end_wl, steps).get_value()
    lower_bound = torch.clamp(reflective_props_tensor - tolerance / 2, 0, 1)
    upper_bound = torch.clamp(reflective_props_tensor + tolerance / 2, 0, 1)
    random_pattern = ReflectivePropsPattern(start_wl, end_wl, lower_bound, upper_bound)

    model = GradientPredictor(num_layers)
    prediction = model.predict(random_pattern)
    thicknesses = prediction.get_thicknesses()
    print(f"Predicted thicknesses: {thicknesses.numpy()}")