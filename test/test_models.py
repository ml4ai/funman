import glob
import json
import os
import unittest


from funman.model import Model

RESOURCES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../resources"
)

ensemble_files = glob.glob(
    os.path.join(RESOURCES, "miranet", "ensemble", "*_miranet.json")
)

models = [
    os.path.join(RESOURCES, "bilayer", "SIDARTHE_BiLayer.json"),
    os.path.join(RESOURCES, "common_model", "petrinet", "sir.json"),
    os.path.join(RESOURCES, "common_model", "petrinet", "sir_typed.json"),
    os.path.join(RESOURCES, "miranet", "ensemble", "BIOMD0000000955_miranet.json"),
    os.path.join(RESOURCES, "common_model", "regnet", "lotka_voltera.json"),
    
]



class TestModels(unittest.TestCase):
    def test_models(self):
        for model in models:
            self.run_instance(model)

    def funman_models(self):
        # import funman.model
        # model_classes = [c[1] for c in inspect.getmembers(funman.model, inspect.isclass)]
        # model_classes = [obj for name, obj in inspect.getmembers(module) if inspect.isclass(obj)]
        model_classes = Model.__subclasses__()
        return model_classes


    def run_instance(self, model_file: str):
        with open(model_file, "r") as f:
            model_json = json.load(f)
        model_classes = self.funman_models()
        model = None
        for model_class in model_classes:
            try:
                model = model_class.from_dict(model_json)
                break
            except Exception:
                pass 
        if model is not None:
            print(f"Created a model of type {type(model)} from model file: {model_file}")
        else:
            print(f"Could not create a model of any type from model file: {model_file}")
        assert(model)       



if __name__ == "__main__":
    unittest.main()
