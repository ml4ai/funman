import json
import os
import unittest

from funman_demo.handlers import RealtimeResultPlotter, ResultCacheWriter
from funman_demo.sim.CHIME.CHIME_SIR import main as run_CHIME_SIR
from pysmt.shortcuts import GE, LE, And, Real, Symbol
from pysmt.typing import REAL

from funman import (
    BilayerDynamics,
    BilayerModel,
    ConsistencyScenario,
    ConsistencyScenarioResult,
    EncodedModel,
    Funman,
    FUNMANConfig,
    ModelParameter,
    ParameterSynthesisScenario,
    ParameterSynthesisScenarioResult,
    QueryFunction,
    QueryLE,
    QueryTrue,
    ResultCombinedHandler,
    SimulationScenario,
    SimulationScenarioResult,
    SimulatorModel,
    StructureParameter,
)

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

        query = QueryLE(variable="I", ub=infected_threshold)

        funman = Funman()

        sim_result: SimulationScenarioResult = funman.solve(
            SimulationScenario(
                model=SimulatorModel(main_fn=run_CHIME_SIR),
                query=QueryFunction(function=does_not_cross_threshold),
            )
        )

        with open(bilayer_path, "r") as f:
            bilayer_src = json.load(f)

        consistency_result: ConsistencyScenarioResult = funman.solve(
            ConsistencyScenario(
                model=BilayerModel(
                    bilayer=BilayerDynamics.from_json(bilayer_src=bilayer_src),
                    init_values=init_values,
                ),
                query=query,
                parameters=[
                    StructureParameter(name="num_steps", lb=3, ub=3),
                    StructureParameter(name="step_size", lb=1, ub=1),
                ],
                config=FUNMANConfig(
                    solver="dreal",
                    dreal_mcts=True,
                    number_of_processes=1,
                    normalize=False,
                ),
            )
        )

        # assert the both queries returned the same result
        return sim_result.query_satisfied == (
            consistency_result.consistent is not None
        )

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
        parameters = [
            ModelParameter(name="x"),
            ModelParameter(name="y"),
        ]
        x = parameters[0].symbol()
        y = parameters[1].symbol()

        formula = And(
            LE(x, Real(5.0)),
            GE(x, Real(0.0)),
            LE(y, Real(12.0)),
            GE(y, Real(10.0)),
        )

        funman = Funman()
        model = EncodedModel(parameters=parameters)
        model._formula = formula
        result: ParameterSynthesisScenarioResult = funman.solve(
            ParameterSynthesisScenario(
                parameters=parameters,
                model=model,
                query=QueryTrue(),
            ),
            config=FUNMANConfig(
                solver="dreal",
                dreal_mcts=True,
                tolerance=1e-8,
                number_of_processes=1,
                normalize=False,
                simplify_query=False,
                normalization_constant=12,
            ),
        )
        assert result

    def setup_use_case_bilayer_common(self):
        bilayer_path = os.path.join(
            RESOURCES, "bilayer", "CHIME_SIR_dynamics_BiLayer.json"
        )
        with open(bilayer_path, "r") as f:
            bilayer_src = json.load(f)

        infected_threshold = 3
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
        )

        query = QueryLE(variable="I", ub=infected_threshold)

        return model, query

    def setup_use_case_bilayer_parameter_synthesis(self):
        model, query = self.setup_use_case_bilayer_common()
        [lb, ub] = model.parameter_bounds["beta"]
        scenario = ParameterSynthesisScenario(
            parameters=[
                ModelParameter(name="beta", lb=lb, ub=ub),
                StructureParameter(name="num_steps", lb=3, ub=3),
                StructureParameter(name="step_size", lb=1, ub=1),
            ],
            model=model,
            query=query,
        )

        return scenario

    def test_use_case_bilayer_parameter_synthesis(self):
        scenario = self.setup_use_case_bilayer_parameter_synthesis()
        funman = Funman()
        result: ParameterSynthesisScenarioResult = funman.solve(
            scenario,
            config=FUNMANConfig(
                # solver="dreal",
                # dreal_mcts=True,
                tolerance=1e-6,
                number_of_processes=1,
                normalize=False,
                simplify_query=False,
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

        scenario = ConsistencyScenario(
            model=model,
            query=query,
            parameters=[
                StructureParameter(name="num_steps", lb=3, ub=3),
                StructureParameter(name="step_size", lb=1, ub=1),
            ],
        )
        return scenario

    def test_use_case_bilayer_consistency(self):
        scenario = self.setup_use_case_bilayer_consistency()

        funman = Funman()

        # Show that region in parameter space is sat (i.e., there exists a true point)
        result_sat: ConsistencyScenarioResult = funman.solve(
            scenario,
            config=FUNMANConfig(
                solver="dreal", normalize=False, simplify_query=False
            ),
        )
        df = result_sat.dataframe(result_sat.parameter_space.true_points[0])

        assert abs(df["I"][2] - 2.0) < 1.0
        beta = result_sat._parameters(
            result_sat.parameter_space.true_points[0]
        )["beta"]
        assert abs(beta - 0.00005) < 0.001

        # Show that region in parameter space is unsat/false
        scenario = self.setup_use_case_bilayer_consistency()
        scenario.model.parameter_bounds["beta"] = [
            0.000067 * 1.5,
            0.000067 * 1.75,
        ]
        result_unsat: ConsistencyScenarioResult = funman.solve(
            scenario, config=FUNMANConfig(solver="dreal", normalize=False)
        )
        assert not result_unsat.consistent


if __name__ == "__main__":
    unittest.main()
