import glob
import json
import os
import unittest

import matplotlib.pyplot as plt
from funman_demo.handlers import RealtimeResultPlotter, ResultCacheWriter

from funman import Funman
from funman.funman import FUNMANConfig
from funman.model import RegnetModel
from funman.model.query import QueryGE, QueryLE, QueryTrue
from funman.model.regnet import RegnetDynamics
from funman.representation.representation import ModelParameter
from funman.scenario import (
    ConsistencyScenario,
    ConsistencyScenarioResult,
    ParameterSynthesisScenario,
    ParameterSynthesisScenarioResult,
)
from funman.utils.handlers import ResultCombinedHandler

RESOURCES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../resources"
)

regnet_files = glob.glob(
    os.path.join(RESOURCES, "common_model", "regnet", "*.json")
)


class TestUseCases(unittest.TestCase):
    def setup_use_case_regnet_common(self):
        regnet_path = regnet_files[0]
        with open(regnet_path, "r") as f:
            regnet_src = json.load(f)

        init_values = {}

        model = RegnetModel(
            regnet=RegnetDynamics(json_graph=regnet_src),
            init_values=init_values,
            parameter_bounds={
                # "predation": [0.51, 0.55],
                # "alpha": [0.78, 0.82],
                # "beta": [1.26, 1.30],
                "predation": [0.8, 0.9],
                "alpha": [0.8, 0.9],
                "beta": [0.8, 1],
                # "gamma": [0.9, 1.1],
                # "delta": [0.9, 1.1],
            },
            structural_parameter_bounds={
                "num_steps": [2, 2],
                "step_size": [4, 4],
            },
        )

        # query = QueryLE(variable="R", ub=10000)
        # query = QueryTrue()
        query = QueryGE(variable="R", lb=1e-6)

        return model, query

    def setup_use_case_regnet_parameter_synthesis(self):
        model, query = self.setup_use_case_regnet_common()

        scenario = ParameterSynthesisScenario(
            parameters=[
                ModelParameter(name=param, lb=bounds[0], ub=bounds[1])
                for param, bounds in model.parameter_bounds.items()
            ],
            model=model,
            query=query,
        )

        return scenario

    def test_use_case_regnet_parameter_synthesis(self):
        scenario = self.setup_use_case_regnet_parameter_synthesis()
        funman = Funman()
        handler = (
            ResultCombinedHandler(
                [
                    ResultCacheWriter(f"box_search.json"),
                    RealtimeResultPlotter(
                        scenario.parameters,
                        plot_points=True,
                        title=f"Parameter Space",
                        realtime_save_path=f"box_search.png",
                    ),
                ]
            ),
        )
        config = (
            FUNMANConfig(
                solver="dreal",
                dreal_mcts=True,
                tolerance=0.04,
                number_of_processes=1,
                save_smtlib=True,
                dreal_precision=0.01,
                dreal_log_level="info",
                _handler=handler[0],
                num_initial_boxes=8,
            ),
        )
        config[0]._handler = handler[0]
        result: ParameterSynthesisScenarioResult = funman.solve(
            scenario, config=config[0]
        )
        assert len(result.parameter_space.true_boxes) > 0
        assert len(result.parameter_space.false_boxes) > 0

    def setup_use_case_regnet_consistency(self):
        model, query = self.setup_use_case_regnet_common()

        scenario = ConsistencyScenario(model=model, query=query)
        return scenario

    @unittest.skip(reason="tmp")
    def test_use_case_regnet_consistency(self):
        scenario = self.setup_use_case_regnet_consistency()

        # Show that region in parameter space is sat (i.e., there exists a true point)
        result_sat: ConsistencyScenarioResult = Funman().solve(
            scenario, config=FUNMANConfig(solver="dreal", dreal_mcts=True)
        )
        assert result_sat.consistent
        df = result_sat.dataframe()
        print(result_sat._parameters())
        print(df)
        result_sat.plot(variables=scenario.model._state_var_names())
        plt.savefig("regnet.png")

        # assert abs(df["Infected"][2] - 2.24) < 0.13
        beta = result_sat._parameters()["beta"]
        # assert abs(beta - 0.00005) < 0.001


if __name__ == "__main__":
    unittest.main()
