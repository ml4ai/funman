import sys

sys.path.append("..")
import json
import os
import unittest

import matplotlib.pyplot as plt
import pandas as pd
from evaluation23 import TestUseCases
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
from funman.representation.representation import Parameter
from funman.scenario import ConsistencyScenario, ConsistencyScenarioResult
from funman.scenario.parameter_synthesis import ParameterSynthesisScenario
from funman.scenario.scenario import AnalysisScenario
from funman.utils.handlers import ResultCombinedHandler

# from funman_demo.handlers import RealtimeResultPlotter, ResultCacheWriter
# from funman.utils.handlers import ResultCombinedHandler

RESOURCES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../resources"
)


class TestUseCases(TestUseCases):
    def sidarthe_bilayer(self):
        with open(
            os.path.join(RESOURCES, "bilayer", "SIDARTHE_BiLayer.json"),
            "r",
        ) as f:
            bilayer = json.load(f)
        return bilayer

    def initial_state_sidarthe(self):
        with open(
            os.path.join(
                RESOURCES, "evaluation23", "SIDARTHE", "SIDARTHE-ic-unit1.json"
            ),
            "r",
        ) as f:
            return json.load(f)

    def bounds_sidarthe(self):
        with open(
            os.path.join(
                RESOURCES,
                "evaluation23",
                "SIDARTHE",
                "SIDARTHE-params-unit1.json",
            ),
            "r",
        ) as f:
            params = json.load(f)
        return {k: [v, v] for k, v in params.items()}

    def make_bounds(self, steps, init_values, tolerance=1e-5):
        return And(
            [
                And(
                    [
                        LE(
                            Plus(
                                [Symbol(f"{v}_{i}", REAL) for v in init_values]
                            ),
                            Real(1.0 + tolerance),
                        ),
                        GE(
                            Plus(
                                [Symbol(f"{v}_{i}", REAL) for v in init_values]
                            ),
                            Real(1.0 - tolerance),
                        ),
                        And(
                            [
                                GE(Symbol(f"{v}_{i}", REAL), Real(0.0))
                                for v in init_values
                            ]
                        ),
                    ]
                )
                for i in range(steps + 1)
            ]
        )

    def sidarthe_query(self, steps, init_values):
        query = QueryEncoded()
        query._formula = self.make_bounds(steps, init_values)
        print(query._formula)
        return query

    def sidarthe_query_1_1_d_1d(self, steps, init_values, bound=(1.0 / 3.0)):
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
                        for step in range(steps + 1)
                    ]
                )
            ]
        )

        # print(query._formula)
        return query

    def sidarthe_extra_1_1_d_1d(self, steps, init_values):
        return And(
            [
                self.make_bounds(steps, init_values),
                Equals(
                    Symbol("theta", REAL),
                    Times(Real(2.0), Symbol("epsilon", REAL)),
                ),
            ]
        )

    def sidarthe_extra_1_1_d_2d(self, steps, init_values):
        return And(
            [
                self.make_bounds(steps, init_values),
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

    def sidarthe_identical(self):
        return []

    @unittest.skip("tmp")
    def test_scenario_2_a(self):
        self.iteration = 0
        case = {
            "model_fn": self.sidarthe_bilayer,
            "initial": self.initial_state_sidarthe,
            "bounds": self.bounds_sidarthe,
            "identical": self.sidarthe_identical,
            "steps": 20,
            "query": self.sidarthe_query,
            "report": self.report,
            "initial_state_tolerance": 0,
        }
        scenario = self.make_scenario(
            BilayerDynamics(json_graph=case["model_fn"]()),
            case["initial"](),
            case["bounds"](),
            case["identical"](),
            case["steps"],
            case["query"](case["steps"], case["initial"]()),
        )
        config = FUNMANConfig(
            max_steps=case["steps"],
            solver="dreal",
            initial_state_tolerance=case["initial_state_tolerance"],
        )
        result_sat = Funman().solve(scenario, config=config)
        case["report"](result_sat)

    @unittest.skip("tmp")
    def test_scenario_2_1_d_1d(self):
        self.iteration = 0
        case = {
            "model_fn": self.sidarthe_bilayer,
            "initial": self.initial_state_sidarthe,
            "bounds": self.bounds_sidarthe,
            "identical": self.sidarthe_identical,
            "steps": 65,
            "query": self.sidarthe_query_1_1_d_1d,
            "report": self.report,
            "initial_state_tolerance": 0,
            "extra_constraints": self.sidarthe_extra_1_1_d_1d,
        }
        bounds = case["bounds"]()
        # bounds["epsilon"] = [0.170, 0.172]
        epsilon_tolerance = 0.005
        bounds["epsilon"] = [
            bounds["epsilon"][0],
            bounds["epsilon"][1] + epsilon_tolerance,
        ]
        bounds["theta"] = [
            bounds["epsilon"][0] * 2.0,
            bounds["epsilon"][1] * 2.0,
        ]  # theta = 2*epsilon
        scenario = self.make_scenario(
            BilayerDynamics(json_graph=case["model_fn"]()),
            case["initial"](),
            bounds,
            case["identical"](),
            case["steps"],
            case["query"](case["steps"], case["initial"](), bound=0.33),
            extra_constraints=case["extra_constraints"](
                case["steps"], case["initial"]()
            ),
        )
        config = FUNMANConfig(
            max_steps=case["steps"],
            solver="dreal",
            initial_state_tolerance=case["initial_state_tolerance"],
        )
        result_sat = Funman().solve(scenario, config=config)
        case["report"](result_sat)

        case = {
            "model_fn": self.sidarthe_bilayer,
            "initial": self.initial_state_sidarthe,
            "bounds": self.bounds_sidarthe,
            "identical": self.sidarthe_identical,
            "steps": 65,
            "query": self.sidarthe_query_1_1_d_1d,
            "report": self.report,
            "initial_state_tolerance": 0,
            "extra_constraints": self.sidarthe_extra_1_1_d_1d,
        }
        bounds = case["bounds"]()
        epsilon_tolerance = 0.001
        bounds["epsilon"] = [
            bounds["epsilon"][0],
            bounds["epsilon"][1] + epsilon_tolerance,
        ]
        bounds["theta"] = [
            bounds["epsilon"][0] * 2.0,
            bounds["epsilon"][1] * 2.0,
        ]  # theta = 2*epsilon
        scenario = self.make_ps_scenario(
            BilayerDynamics(json_graph=case["model_fn"]()),
            case["initial"](),
            bounds,
            case["identical"](),
            case["steps"],
            case["query"](case["steps"], case["initial"](), bound=0.33),
            params_to_synth=["epsilon"],
            extra_constraints=case["extra_constraints"](
                case["steps"], case["initial"]()
            ),
        )
        config = FUNMANConfig(
            max_steps=case["steps"],
            solver="dreal",
            number_of_processes=1,
            num_initial_boxes=40,
            initial_state_tolerance=case["initial_state_tolerance"],
            tolerance=1e-6,
        )
        config._handler = ResultCombinedHandler(
            [
                ResultCacheWriter(f"box_search.json"),
                RealtimeResultPlotter(
                    scenario.parameters,
                    plot_points=True,
                    title=f"Feasible Regions (epsilon)",
                    realtime_save_path=f"box_search.png",
                    dpi=600,
                ),
            ]
        )
        result_sat = Funman().solve(scenario, config=config)
        # case["report"](result_sat)

    def test_scenario_2_1_d_2d(self):
        self.iteration = 0
        case = {
            "model_fn": self.sidarthe_bilayer,
            "initial": self.initial_state_sidarthe,
            "bounds": self.bounds_sidarthe,
            "identical": self.sidarthe_identical,
            "steps": 65,
            "query": self.sidarthe_query_1_1_d_1d,
            "report": self.report,
            "initial_state_tolerance": 0,
            "extra_constraints": self.sidarthe_extra_1_1_d_2d,
        }
        bounds = case["bounds"]()
        # bounds["epsilon"] = [0.170, 0.172]
        epsilon_tolerance = 0.00001
        # theta_tolerance = 0.001
        bounds["epsilon"] = [
            bounds["epsilon"][0],
            bounds["epsilon"][1] + epsilon_tolerance,
        ]
        bounds["theta"] = [
            2 * bounds["epsilon"][0],
            2 * bounds["epsilon"][1],
        ]
        scenario = self.make_scenario(
            BilayerDynamics(json_graph=case["model_fn"]()),
            case["initial"](),
            bounds,
            case["identical"](),
            case["steps"],
            case["query"](case["steps"], case["initial"](), bound=0.33),
            case["extra_constraints"](case["steps"], case["initial"]()),
        )
        config = FUNMANConfig(
            max_steps=case["steps"],
            solver="dreal",
            initial_state_tolerance=case["initial_state_tolerance"],
        )
        # result_sat = Funman().solve(scenario, config=config)
        # case["report"](result_sat)

        case = {
            "model_fn": self.sidarthe_bilayer,
            "initial": self.initial_state_sidarthe,
            "bounds": self.bounds_sidarthe,
            "identical": self.sidarthe_identical,
            "steps": 60,
            "query": self.sidarthe_query_1_1_d_1d,
            "report": self.report,
            "initial_state_tolerance": 0,
            "extra_constraints": self.sidarthe_extra_1_1_d_2d,
        }
        bounds = case["bounds"]()
        epsilon_tolerance = 1e-9
        # theta_tolerance = 0.0005
        bounds["epsilon"] = [
            bounds["epsilon"][0],
            bounds["epsilon"][1] + epsilon_tolerance,
        ]
        bounds["theta"] = [
            2 * bounds["epsilon"][0],
            2 * bounds["epsilon"][1],
        ]
        scenario = self.make_ps_scenario(
            BilayerDynamics(json_graph=case["model_fn"]()),
            case["initial"](),
            bounds,
            case["identical"](),
            case["steps"],
            case["query"](case["steps"], case["initial"](), bound=0.33),
            params_to_synth=["epsilon", "theta"],
            extra_constraints=case["extra_constraints"](
                case["steps"], case["initial"]()
            ),
        )
        config = FUNMANConfig(
            max_steps=case["steps"],
            solver="dreal",
            number_of_processes=1,
            num_initial_boxes=1,
            initial_state_tolerance=case["initial_state_tolerance"],
            tolerance=1e-11,
            save_smtlib=True,
        )
        config._handler = ResultCombinedHandler(
            [
                # ResultCacheWriter(f"box_search.json"),
                RealtimeResultPlotter(
                    scenario.parameters,
                    plot_points=True,
                    title=f"Feasible Regions (epsilon)",
                    realtime_save_path=f"box_search.png",
                    dpi=600,
                ),
            ]
        )
        result_sat = Funman().solve(scenario, config=config)
        # case["report"](result_sat)


if __name__ == "__main__":
    unittest.main()
