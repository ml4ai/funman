import json
import os
import unittest

from funman_demo.handlers import RealtimeResultPlotter, ResultCacheWriter

from funman.funman import Funman, FUNMANConfig
from funman.model import Model
from funman.model.generated_models.petrinet import Model as GeneratedPetrinet
from funman.model.generated_models.regnet import Model as GeneratedRegnet
from funman.model.petrinet import GeneratedPetriNetModel
from funman.model.query import QueryTrue
from funman.model.regnet import GeneratedRegnetModel
from funman.representation.representation import Parameter
from funman.scenario.parameter_synthesis import (
    ParameterSynthesisScenario,
    ParameterSynthesisScenarioResult,
)
from funman.utils.handlers import ResultCombinedHandler

RESOURCES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../resources"
)

models = {
    os.path.join(
        RESOURCES, "common_model", "petrinet", "sir.json"
    ): GeneratedPetrinet,
    os.path.join(
        RESOURCES, "common_model", "petrinet", "sir_typed.json"
    ): GeneratedPetrinet,
    os.path.join(
        RESOURCES, "common_model", "regnet", "lotka_volterra.json"
    ): GeneratedRegnet,
    os.path.join(
        RESOURCES, "common_model", "regnet", "syntax_edge_cases.json"
    ): GeneratedRegnet,
}


class TestModels(unittest.TestCase):
    def test_models(self):
        for model in models:
            self.run_instance(model)

    def run_instance(self, model_file: str):
        with open(model_file, "r") as f:
            model_json = json.load(f)
        generated_model = models[model_file].parse_raw(json.dumps(model_json))
        if isinstance(generated_model, GeneratedRegnet):
            model = GeneratedRegnetModel(
                regnet=generated_model,
                structural_parameter_bounds={
                    "num_steps": [2, 2],
                    "step_size": [1, 1],
                },
            )
            parameters = [Parameter(name="beta", lb=lb, ub=ub)]

        elif isinstance(generated_model, GeneratedPetrinet):
            model = GeneratedPetriNetModel(petrinet=generated_model)
            parameters = []
        query = QueryTrue()

        scenario = ParameterSynthesisScenario(
            parameters=parameters,
            model=model,
            query=query,
        )

        result: ParameterSynthesisScenarioResult = Funman().solve(
            scenario,
            config=FUNMANConfig(
                solver="dreal",
                dreal_mcts=True,
                tolerance=1e-8,
                number_of_processes=1,
                save_smtlib=True,
                # dreal_log_level="debug",
                _handler=ResultCombinedHandler(
                    [
                        ResultCacheWriter(f"box_search.json"),
                        RealtimeResultPlotter(
                            scenario.parameters,
                            plot_points=True,
                            title=f"Feasible Regions (beta)",
                            realtime_save_path=f"box_search.png",
                        ),
                    ]
                ),
            ),
        )
        assert result


if __name__ == "__main__":
    unittest.main()
