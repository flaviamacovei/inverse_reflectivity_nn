import torch
from config import tolerance
from forward.forward_tmm import coating_to_reflective_props
from prediction.GradientPredictor import GradientPredictor
from data.values.ReflectivePropsPattern import ReflectivePropsPattern
from data.values.Coating import Coating
from ui.visualise import visualise
from ui.FileInput import FileInput
from data.values.RefractiveIndex import RefractiveIndex

if __name__ == "__main__":
    num_layers = 10

    file_input = FileInput()
    file_input.read_from_csv("data/Neuschwanstein_target.csv")
    pattern = file_input.to_reflective_props_pattern()
    visualise(refs = pattern, filename = "original")

    model = GradientPredictor(num_layers)
    prediction = model.predict(pattern)
    thicknesses = prediction.get_thicknesses()
    refractive_indices = prediction.get_refractive_indices()

    print(f"Predicted thicknesses: {thicknesses.detach().numpy()}")
    print(f"Predicted refractive indices: {refractive_indices.detach().numpy()}")


    refractive_indices_rounded = torch.tensor([RefractiveIndex.round(x) for x in refractive_indices])
    true_coating = Coating(thicknesses, refractive_indices_rounded)



    print(f"Predicted refractive indices: {refractive_indices_rounded.detach().numpy()}")

    optimal_reflective_props = coating_to_reflective_props(true_coating)
    visualise(optimal_reflective_props, pattern, "optimised")