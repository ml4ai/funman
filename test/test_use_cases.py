import os
import unittest

from funman_demo.handlers import RealtimeResultPlotter, ResultCacheWriter
from funman_demo.sim.CHIME.CHIME_SIR import main as run_CHIME_SIR
from pysmt.shortcuts import GE, LE, And, Real, Symbol
from pysmt.typing import REAL

from funman import Funman
from funman.model import (
    EncodedModel,
    Parameter,
    QueryFunction,
    QueryLE,
    QueryTrue,
    SimulatorModel,
)
from funman.model.bilayer import BilayerDynamics, BilayerModel
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
from funman.search import SearchConfig
from funman.search.handlers import ResultCombinedHandler

RESOURCES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../resources"
)


class TestUseCases(unittest.TestCase):
    def compare_against_CHIME_Sim(
        self, bilayer_path, init_values, infected_threshold
    ):
        # query the simulator
        def does_not_cross_threshold(sim_results):
            i = sim_results[2]
            return all(i_t <= infected_threshold for i_t in i)

        query = QueryLE("I", infected_threshold)

        funman = Funman()

        sim_result: SimulationScenarioResult = funman.solve(
            SimulationScenario(
                model=SimulatorModel(run_CHIME_SIR),
                query=QueryFunction(does_not_cross_threshold),
            )
        )

        consistency_result: ConsistencyScenarioResult = funman.solve(
            ConsistencyScenario(
                model=BilayerModel(
                    BilayerDynamics.from_json(bilayer_path),
                    init_values=init_values,
                ),
                query=query,
            )
        )

        # assert the both queries returned the same result
        return sim_result.query_satisfied == consistency_result.query_satisfied

    def test_use_case_bilayer_consistency(self):
        """
        This test compares a BilayerModel against a SimulatorModel to determine whether their response to a query is identical.
        """
        bilayer_path = os.path.join(
            RESOURCES, "bilayer", "CHIME_SIR_dynamics_BiLayer.json"
        )
        infected_threshold = 130
        init_values = {"S": 9998, "I": 1, "R": 1}
        assert self.compare_against_CHIME_Sim(
            bilayer_path, init_values, infected_threshold
        )

    def test_use_case_simple_parameter_synthesis(self):
        x = Symbol("x", REAL)
        y = Symbol("y", REAL)

        formula = And(
            LE(x, Real(5.0)),
            GE(x, Real(0.0)),
            LE(y, Real(12.0)),
            GE(y, Real(10.0)),
        )

        funman = Funman()
        result: ParameterSynthesisScenarioResult = funman.solve(
            ParameterSynthesisScenario(
                [
                    Parameter("x", _symbol=x),
                    Parameter("y", _symbol=y),
                ],
                EncodedModel(formula),
            )
        )
        assert result

    def setup_use_case_bilayer_common(self):
        bilayer_path = os.path.join(
            RESOURCES, "bilayer", "CHIME_SIR_dynamics_BiLayer.json"
        )
        infected_threshold = 3
        init_values = {"S": 9998, "I": 1, "R": 1}

        scale_factor = 0.75
        lb = 0.000067 * (1 - scale_factor)
        ub = 0.000067 * (1 + scale_factor)

        model = BilayerModel(
            BilayerDynamics.from_json(bilayer_path),
            init_values=init_values,
            parameter_bounds={
                "beta": [lb, ub],
                "gamma": [1.0 / 14.0, 1.0 / 14.0],
            },
        )

        query = QueryLE("I", infected_threshold)

        return model, query

    def setup_use_case_bilayer_parameter_synthesis(self):
        model, query = self.setup_use_case_bilayer_common()
        [lb, ub] = model.parameter_bounds["beta"]
        scenario = ParameterSynthesisScenario(
            parameters=[Parameter("beta", lb=lb, ub=ub)],
            model=model,
            query=query,
        )

        return scenario

    def test_use_case_bilayer_parameter_synthesis(self):
        scenario = self.setup_use_case_bilayer_parameter_synthesis()
        funman = Funman()
        result: ParameterSynthesisScenarioResult = funman.solve(
            scenario,
            config=SearchConfig(
                tolerance=1e-8,
                number_of_processes=1,
                handler=ResultCombinedHandler(
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
            config=FUNMANConfig(tolerance=1e-8),
        )
        assert len(result.parameter_space.true_boxes) > 0
        assert len(result.parameter_space.false_boxes) > 0

    def setup_use_case_bilayer_consistency(self):
        model, query = self.setup_use_case_bilayer_common()

        scenario = ConsistencyScenario(model=model, query=query)
        return scenario

    def test_use_case_bilayer_consistency(self):
        scenario = self.setup_use_case_bilayer_consistency()

        funman = Funman()

        # Show that region in parameter space is sat (i.e., there exists a true point)
        result_sat: ConsistencyScenarioResult = funman.solve(scenario)
        df = result_sat.dataframe()

        assert abs(df["I"][2] - 2.04) < 0.01
        assert abs(df["beta"][0] - 0.00005) < 0.000001

        # Show that region in parameter space is unsat/false
        scenario.model.parameter_bounds["beta"] = [
            0.000067 * 1.5,
            0.000067 * 1.75,
        ]
        result_unsat: ConsistencyScenarioResult = funman.solve(scenario)
        assert not result_unsat.consistent


if __name__ == "__main__":
    unittest.main()
