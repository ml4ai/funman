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
from funman.representation.representation import ModelParameter
from funman.scenario import (
    ConsistencyScenario,
    ConsistencyScenarioResult,
    ParameterSynthesisScenario,
    ParameterSynthesisScenarioResult,
)
from funman.scenario.simulation import (
    SimulationScenario,
    SimulationScenarioResult,
)
from funman.utils.handlers import ResultCombinedHandler

RESOURCES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../resources"
)


class TestUseCases(unittest.TestCase):
    def setup_use_case_bilayer_common(self):
        bilayer_path = os.path.join(
            RESOURCES, "bilayer", "CHIME_SIR_dynamics_BiLayer.json"
        )
        with open(bilayer_path, "r") as f:
            bilayer_src = json.load(f)

        infected_threshold = 1000
        init_values = {"S": 9998, "I": 1, "R": 1}

        scale_factor = 0.5
        lb = 0.000067 * (1 - scale_factor)
        ub = 0.000067 * (1 + scale_factor)

        model = BilayerModel(
            bilayer=BilayerDynamics(json_graph=bilayer_src),
            init_values=init_values,
            parameter_bounds={
                "beta": [lb, ub],
                "gamma": [1.0 / 14.0, 1.0 / 14.0],
            },
            structural_parameter_bounds={
                "num_steps": [100, 100],
                "step_size": [1, 1],
            },
        )

        query = QueryLE(variable="I", ub=infected_threshold, at_end=True)

        return model, query

    def setup_use_case_bilayer_parameter_synthesis(self):
        model, query = self.setup_use_case_bilayer_common()
        [lb, ub] = model.parameter_bounds["beta"]
        scenario = ParameterSynthesisScenario(
            parameters=[ModelParameter(name="beta", lb=lb, ub=ub)],
            model=model,
            query=query,
        )

        return scenario

    @unittest.skip(reason="tmp")
    def test_use_case_bilayer_parameter_synthesis(self):
        scenario = self.setup_use_case_bilayer_parameter_synthesis()
        funman = Funman()
        result: ParameterSynthesisScenarioResult = funman.solve(
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
        assert len(result.parameter_space.true_boxes) > 0
        assert len(result.parameter_space.false_boxes) > 0

    def setup_use_case_bilayer_consistency(self):
        model, query = self.setup_use_case_bilayer_common()

        scenario = ConsistencyScenario(model=model, query=query)
        return scenario

    def test_use_case_bilayer_consistency(self):
        scenario = self.setup_use_case_bilayer_consistency()

        # Show that region in parameter space is sat (i.e., there exists a true point)
        result_sat: ConsistencyScenarioResult = Funman().solve(
            scenario,
            config=FUNMANConfig(
                solver="dreal",
                dreal_mcts=True,
                tolerance=1e-8,
                number_of_processes=1,
                save_smtlib=True,
            ),
        )
        df = result_sat.dataframe()

        assert abs(df["I"][2] - 2.24) < 0.13
        beta = result_sat._parameters()["beta"]
        assert abs(beta - 0.00005) < 0.001

        scenario = self.setup_use_case_bilayer_consistency()
        # Show that region in parameter space is unsat/false
        scenario.model.parameter_bounds["beta"] = [
            0.000067 * 1.5,
            0.000067 * 1.75,
        ]
        result_unsat: ConsistencyScenarioResult = Funman().solve(scenario)
        assert not result_unsat.consistent


if __name__ == "__main__":
    unittest.main()
