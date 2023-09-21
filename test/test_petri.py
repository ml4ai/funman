import glob
import json
import os
import unittest

import matplotlib.pyplot as plt
from funman_demo.handlers import RealtimeResultPlotter, ResultCacheWriter

from funman import Funman
from funman.config import FUNMANConfig
from funman.model import EnsembleModel, PetrinetModel, QueryLE
from funman.model.petrinet import PetrinetDynamics
from funman.model.query import QueryAnd
from funman.representation.representation import ModelParameter
from funman.representation.symbol import ModelSymbol
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

ensemble_files = glob.glob(
    os.path.join(RESOURCES, "miranet", "ensemble", "*_miranet.json")
)


class TestUseCases(unittest.TestCase):
    def setup_use_case_petri_ensemble_common(self):
        m1, q1 = self.setup_use_case_petri_common(name="m1")
        m2, q2 = self.setup_use_case_petri_common(name="m2")
        m = EnsembleModel(models=[m1, m2])
        return m, QueryAnd(queries=[q1, q2])

    def setup_use_case_petri_common(self, name="m"):
        petri_path = ensemble_files[0]
        with open(petri_path, "r") as f:
            petri_src = json.load(f)

        infected_threshold = 0.3

        N0 = 60e6
        I0, D0, A0, R0, T0, H0, E0 = 200 / N0, 20 / N0, 1 / N0, 2 / N0, 0, 0, 0
        S0 = 1 - I0 - D0 - A0 - R0 - T0 - H0 - E0
        C0 = 0
        init_values = {
            "Susceptible": S0,
            "Infected": I0,
            "Diagnosed": D0,
            "Ailing": A0,
            "Recognized": R0,
            "Healed": H0,
            "Threatened": T0,
            "Extinct": E0,
            "Cases": C0,
            "Hospitalizations": H0,
            "Deaths": D0,
        }

        scale_factor = 0.5
        lb = 0.000067 * (1 - scale_factor)
        ub = 0.000067 * (1 + scale_factor)

        model = PetrinetModel(
            name=name,
            petrinet=PetrinetDynamics(json_graph=petri_src),
            init_values=init_values,
            parameter_bounds={
                "beta": [lb, ub],
                "gamma": [1.0 / 14.0, 1.0 / 14.0],
            },
            structural_parameter_bounds={
                "num_steps": [50, 50],
                "step_size": [1, 1],
            },
        )

        query = QueryLE(
            variable=ModelSymbol(name="Infected", model=model.name),
            ub=infected_threshold,
        )

        return model, query

    @unittest.skip(reason="tmp")
    def setup_use_case_petri_parameter_synthesis(self):
        model, query = self.setup_use_case_petri_common()
        [lb, ub] = model.parameter_bounds["beta"]
        scenario = ParameterSynthesisScenario(
            parameters=[ModelParameter(name="beta", lb=lb, ub=ub)],
            model=model,
            query=query,
        )

        return scenario

    def test_use_case_petri_parameter_synthesis(self):
        scenario = self.setup_use_case_petri_parameter_synthesis()
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

    def setup_use_case_petri_consistency(self):
        model, query = self.setup_use_case_petri_common()

        scenario = ConsistencyScenario(model=model, query=query)
        return scenario

    def setup_use_case_petri_ensemble_consistency(self):
        model, query = self.setup_use_case_petri_ensemble_common()

        scenario = ConsistencyScenario(model=model, query=query, parameters=[])
        return scenario

    @unittest.skip("tmp")
    def test_use_case_petri_consistency(self):
        scenario = self.setup_use_case_petri_consistency()

        # Show that region in parameter space is sat (i.e., there exists a true point)
        result_sat: ConsistencyScenarioResult = Funman().solve(scenario)
        assert result_sat.consistent
        df = result_sat.dataframe()

        result_sat.plot(variables=scenario.model._state_var_names())
        plt.savefig("petri.png")

        # assert abs(df["Infected"][2] - 2.24) < 0.13
        beta = result_sat._parameters()["beta"]
        # assert abs(beta - 0.00005) < 0.001

        # scenario = self.setup_use_case_petri_consistency()
        # Show that region in parameter space is unsat/false
        # scenario.model.parameter_bounds["beta"] = [
        #     0.000067 * 1.5,
        #     0.000067 * 1.75,
        # ]
        # result_unsat: ConsistencyScenarioResult = Funman().solve(scenario)
        # assert not result_unsat.consistent

    def test_ensemble(self):
        scenario = self.setup_use_case_petri_ensemble_consistency()

        # Show that region in parameter space is sat (i.e., there exists a true point)
        result_sat: ConsistencyScenarioResult = Funman().solve(scenario)
        assert result_sat.consistent
        point = result_sat.parameter_space.true_points[0]
        df = result_sat.dataframe(point)

        result_sat.plot(
            variables=scenario.model._state_var_names(), point=point
        )
        plt.savefig("petri.png")

        # assert abs(df["Infected"][2] - 2.24) < 0.13
        # beta = result_sat._parameters()["beta"]


if __name__ == "__main__":
    unittest.main()
