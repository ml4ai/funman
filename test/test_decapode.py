import json
import os
import unittest

from funman_demo.handlers import RealtimeResultPlotter, ResultCacheWriter
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
from funman.model.decapode import DecapodeDynamics, DecapodeModel
from funman.representation.representation import Parameter
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
    def setup_use_case_decapode_common(self):
        decapode_path = os.path.join(
            RESOURCES, "decapode", "hydrostatic_3_6.json"
        )
        with open(decapode_path, "r") as f:
            decapode_src = json.load(f)

        geopotential_threshold = 5000
        init_values = {"H": 1}

        scale_factor = 0.75
        lb = 1.67e-27 * (1 - scale_factor)
        ub = 1.67e-27 * (1 + scale_factor)

        model = DecapodeModel(
            decapode=DecapodeDynamics(json_graph=decapode_src),
            init_values=init_values,
            parameter_bounds={
                'R^Mo(Other("*"))': [lb, ub],
                "T_n": [lb, ub],
                'm_Mo(Other("‾"))': [lb, ub],
                "g": [9.8, 9.8],
            },
        )

        query = QueryLE(variable="H", ub=geopotential_threshold)

        return model, query

    def setup_use_case_decapode_parameter_synthesis(self):
        model, query = self.setup_use_case_decapode_common()
        [lb, ub] = model.parameter_bounds['m_Mo(Other("‾"))']
        scenario = ParameterSynthesisScenario(
            parameters=[Parameter(name='m_Mo(Other("‾"))', lb=lb, ub=ub)],
            model=model,
            query=query,
        )

        return scenario

    def test_use_case_decapode_parameter_synthesis(self):
        scenario = self.setup_use_case_decapode_parameter_synthesis()
        funman = Funman()
        result: ParameterSynthesisScenarioResult = funman.solve(
            scenario,
            config=FUNMANConfig(
                tolerance=1e-8,
                number_of_processes=1,
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

    def setup_use_case_decapode_consistency(self):
        model, query = self.setup_use_case_decapode_common()

        scenario = ConsistencyScenario(model=model, query=query)
        return scenario

    def test_use_case_decapode_consistency(self):
        scenario = self.setup_use_case_decapode_consistency()

        funman = Funman()

        # Show that region in parameter space is sat (i.e., there exists a true point)
        result_sat: ConsistencyScenarioResult = funman.solve(scenario)
        df = result_sat.dataframe()

        assert abs(df["H"][2] - 5000) < 0.01

        # Show that region in parameter space is unsat/false
        scenario.model.parameter_bounds['m_Mo(Other("‾"))'] = [
            1.67e-27 * 1.5,
            1.67e-27 * 1.75,
        ]
        result_unsat: ConsistencyScenarioResult = funman.solve(scenario)
        assert not result_unsat.consistent


if __name__ == "__main__":
    unittest.main()
