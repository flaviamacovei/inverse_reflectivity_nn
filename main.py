import torch
from config import tolerance
from forward.forward_tmm import coating_to_reflective_props
from prediction.GradientPredictor import GradientPredictor
from data.values.ReflectivePropsPattern import ReflectivePropsPattern
from data.values.Coating import Coating
from ui.visualise import visualise

if __name__ == "__main__":
    num_layers = 10

    # make random pattern
    thicknesses = torch.rand((num_layers // 2)) * 1.0e-5
    thicknesses[0] = 1
    thicknesses[-1] = 1
    refractive_indices = torch.rand((num_layers // 2))
    refractive_indices.requires_grad_()
    random_coating = Coating(thicknesses, refractive_indices)

    reflective_props = coating_to_reflective_props(random_coating)
    reflective_props_tensor = reflective_props.get_value()
    lower_bound = torch.clamp(reflective_props_tensor - tolerance / 2, 0, 1)
    upper_bound = torch.clamp(reflective_props_tensor + tolerance / 2, 0, 1)
    random_pattern = ReflectivePropsPattern(lower_bound, upper_bound)
    visualise(preds = reflective_props, refs = random_pattern, filename = "original")

    model = GradientPredictor(num_layers)
    prediction = model.predict(random_pattern)
    thicknesses = prediction.get_thicknesses()
    refractive_indices = prediction.get_refractive_indices()
    print(f"Predicted thicknesses: {thicknesses.detach().numpy()}")
    print(f"Predicted refractive indices. {refractive_indices.detach().numpy()}")

    optimal_reflective_props = coating_to_reflective_props(prediction)
    visualise(optimal_reflective_props, random_pattern, "optimised")