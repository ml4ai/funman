import sys

sys.path.append("..")
import unittest

import matplotlib.pyplot as plt
import pandas as pd
from common import TestUnitTests
from funman_demo.handlers import RealtimeResultPlotter, ResultCacheWriter
from pysmt.shortcuts import (
    GE,
    GT,
    LE,
    LT,
    REAL,
    TRUE,
    And,
    Equals,
    Minus,
    Or,
    Plus,
    Real,
    Symbol,
    Times,
)

from funman import Funman
from funman.funman import FUNMANConfig

# from funman.funman import FUNMANConfig
from funman.model import QueryLE
from funman.model.bilayer import BilayerDynamics, BilayerGraph, BilayerModel
from funman.model.query import QueryEncoded, QueryTrue
from funman.representation.representation import ModelParameter
from funman.scenario import ConsistencyScenario, ConsistencyScenarioResult
from funman.scenario.parameter_synthesis import ParameterSynthesisScenario
from funman.scenario.scenario import AnalysisScenario
from funman.utils.handlers import ResultCombinedHandler


class TestUseCases(TestUnitTests):
    steps = 96
    step_size = 12

    s2_models = [
        "Mosaphir_petri_to_bilayer",
        "UF_petri_to_bilayer",
        "Morrison_bilayer",
        "Skema_bilayer",
    ]

    def test_model_0(self):
        self.common_test_model(self.s2_models[0])

    def common_test_model(self, model_name: str):
        result = self.analyze_model(model_name)

    def sidarthe_query_2_1_d(
        self, steps, init_values, bound=(1.0 / 3.0), step_size=1
    ):
        query = QueryEncoded()
        query._formula = And(
            [
                # self.make_bounds(steps, init_values),
                And(
                    [
                        LE(
                            Plus(
                                [
                                    Symbol(f"I_{step}", REAL),
                                    Symbol(f"D_{step}", REAL),
                                    Symbol(f"A_{step}", REAL),
                                    Symbol(f"R_{step}", REAL),
                                    Symbol(f"T_{step}", REAL),
                                ]
                            ),
                            Real(bound),
                        )
                        for step in range(0, steps + 1, step_size)
                    ]
                )
            ]
        )

        # print(query._formula)
        return query

    def sidarthe_extra_2_1_d(self, steps, init_values, step_size=1):
        return And(
            [
                self.make_bounds(steps, init_values, step_size=step_size),
                GE(
                    Symbol("theta", REAL),
                    Times(Real(2.0), Symbol("epsilon", REAL)),
                ),
                # self.make_monotone_constraints(
                #     steps,
                #     init_values=init_values,
                #     var_directions={("S", "decrease"), ("R", "increase")},
                # ),
            ]
        )

    def analyze_model(self, model_name: str):
        initial = self.initial_state_sidarthe()
        bounds = self.bounds_sidarthe()
        bounds["epsilon"] = [0.09, 0.175]
        bounds["theta"] = [0.2, 0.3]
        query = self.sidarthe_query_2_1_d(
            self.steps, initial, bound=0.33, step_size=self.step_size
        )
        extra_constraints = self.sidarthe_extra_2_1_d(
            self.steps, initial, step_size=self.step_size
        )

        bilayer = BilayerDynamics(
            json_graph=self.sidarthe_bilayer(self.models[model_name])
        )
        scenario = self.make_scenario(
            bilayer,
            initial,
            bounds,
            [],
            self.steps,
            query,
            extra_constraints=extra_constraints,
        )
        config = FUNMANConfig(
            num_steps=self.steps,
            step_size=self.step_size,
            solver="dreal",
            initial_state_tolerance=0.0,
        )
        result_sat = Funman().solve(scenario, config=config)
        self.report(result_sat, name=model_name)

        ps_scenario = self.make_ps_scenario(
            bilayer,
            initial,
            bounds,
            [],
            self.steps,
            query,
            extra_constraints=extra_constraints,
            params_to_synth=["epsilon", "theta"],
        )
        config = FUNMANConfig(
            num_steps=self.steps,
            solver="dreal",
            number_of_processes=1,
            num_initial_boxes=128,
            initial_state_tolerance=0.0,
            tolerance=1e-4,
            save_smtlib=False,
            step_size=self.step_size,
        )
        config._handler = ResultCombinedHandler(
            [
                # ResultCacheWriter(f"box_search.json"),
                RealtimeResultPlotter(
                    ps_scenario.parameters,
                    plot_points=True,
                    title=f"Feasible Regions (epsilon)",
                    realtime_save_path=f"box_search.png",
                    dpi=600,
                ),
            ]
        )
        result_sat = Funman().solve(ps_scenario, config=config)
        with open("funman_s2_1_d_parameter_space_.json", "w") as f:
            f.write(result_sat.parameter_space.json())


if __name__ == "__main__":
    unittest.main()
