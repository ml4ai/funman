import json
import os
import unittest

from funman_demo.handlers import RealtimeResultPlotter, ResultCacheWriter
from funman_demo.sim.CHIME.CHIME_SIR import main as run_CHIME_SIR
from pysmt.shortcuts import GE, LE, And, Real, Symbol
from pysmt.typing import REAL

from funman import Funman
from funman.funman import FUNMANConfig
from funman.model import (
    EncodedModel,
    QueryFunction,
    QueryLE,
    QueryTrue,
    SimulatorModel,
)
from funman.model.bilayer import BilayerDynamics, BilayerModel
from funman.representation.representation import Parameter
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


class TestBucky(unittest.TestCase):
    def setup_use_case_bilayer_common(self):
        bilayer_path = os.path.join(
            RESOURCES, "bilayer", "Bucky_SEIIIRRD_BiLayer_v3.json"
        )
        with open(bilayer_path, "r") as f:
            bilayer_src = json.load(f)

        infected_threshold = 50
        init_values = {
            "S": 99.43,
            "E": 0.4,
            "I_asym": 0.1,
            "I_mild": 0.05,
            "I_crit": 0.02,
            "R": 0,
            "R_hosp": 0,
            "D": 0,
        }

        identical_parameters = [
            ["beta_1", "beta_2"],
            ["gamma_1", "gamma_2", "gamma_h"],
        ]

        model = BilayerModel(
            bilayer=BilayerDynamics(json_graph=bilayer_src),
            init_values=init_values,
            # identical_parameters=identical_parameters,
            parameter_bounds={
                "beta_1": [0.0, 1.0],
                "beta_2": [0.0, 1.0],
                "gamma_1": [0.0, 1.0],
                "gamma_2": [0.0, 1.0],
                "gamma_h": [0.0, 1.0],
                "delta_1": [0.0, 1.0],
                "delta_2": [0.0, 1.0],
                "delta_3": [0.0, 1.0],
                "delta_4": [0.0, 1.0],
                "sigma": [0.0, 1.0],
                "theta": [0.0, 1.0],
            },
        )

        query = QueryLE(variable="I_crit", ub=infected_threshold)

        return model, query

    def setup_use_case_bilayer_consistency(self):
        model, query = self.setup_use_case_bilayer_common()

        scenario = ConsistencyScenario(model=model, query=query)
        return scenario

    def setup_use_case_bilayer_parameter_synthesis(self):
        model, query = self.setup_use_case_bilayer_common()

        def make_parameter(name):
            [lb, ub] = model.parameter_bounds[name]
            return Parameter(name=name, lb=lb, ub=ub)

        scenario = ParameterSynthesisScenario(
            parameters=[
                make_parameter("beta_1"),
                make_parameter("beta_2"),
            ],
            model=model,
            query=query,
        )

        return scenario

    unittest.skip(reason="placeholder")

    def test_use_case_bilayer_parameter_synthesis(self):
        scenario = self.setup_use_case_bilayer_parameter_synthesis()
        funman = Funman()
        result: ParameterSynthesisScenarioResult = funman.solve(
            scenario,
            config=FUNMANConfig(
                tolerance=1e-8,
                number_of_processes=1,
                _handler=ResultCombinedHandler(
                    [
                        ResultCacheWriter(f"bucky_box_search.json"),
                        RealtimeResultPlotter(
                            scenario.parameters,
                            plot_points=True,
                            title=f"Feasible Regions (beta)",
                            realtime_save_path=f"bucky_box_search.png",
                        ),
                    ]
                ),
            ),
        )
        assert len(result.parameter_space.true_boxes) > 0
        assert len(result.parameter_space.false_boxes) > 0

    def test_use_case_bilayer_consistency(self):
        scenario = self.setup_use_case_bilayer_consistency()

        funman = Funman()
        config = FUNMANConfig(max_steps=2, step_size=1, solver="dreal")

        # Show that region in parameter space is sat (i.e., there exists a true point)
        result_sat: ConsistencyScenarioResult = funman.solve(
            scenario, config=config
        )
        df = result_sat.dataframe()

        # assert abs(df["I"][2] - 2.24) < 0.01
        # beta = result_sat._parameters()["beta"]
        # assert abs(beta - 0.00005) < 0.00001

        # # Show that region in parameter space is unsat/false
        # scenario.model.parameter_bounds["beta"] = [
        #     0.000067 * 1.5,
        #     0.000067 * 1.75,
        # ]
        # result_unsat: ConsistencyScenarioResult = funman.solve(scenario)
        # assert not result_unsat.consistent


if __name__ == "__main__":
    unittest.main()
