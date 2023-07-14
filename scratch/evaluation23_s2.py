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
from funman.representation.representation import ModelParameter
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
            os.path.join(
                RESOURCES, "bilayer", "SIDARTHE_BiLayer_corrected.json"
            ),
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

    def make_bounds(self, steps, init_values, tolerance=1e-5, step_size=1):
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
                for i in range(0, steps + 1, step_size)
            ]
        )

    def sidarthe_query(self, steps, init_values):
        query = QueryEncoded()
        query._formula = self.make_bounds(steps, init_values)
        print(query._formula)
        return query

    def sidarthe_query_1_1_d_1d(
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

    def sidarthe_extra_1_1_d_2d(self, steps, init_values, step_size=1):
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
            num_steps=case["steps"],
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
            num_steps=case["steps"],
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
            num_steps=case["steps"],
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

    @unittest.skip(reason="tmp")
    def test_scenario_2_1_b_i(self):
        self.iteration = 0
        case = {
            "model_fn": self.sidarthe_bilayer,
            "initial": self.initial_state_sidarthe,
            "bounds": self.bounds_sidarthe,
            "identical": self.sidarthe_identical,
            "steps": 100,
            # "query": self.sidarthe_query_1_1_d_1d,
            "report": self.report,
            "initial_state_tolerance": 0,
            "step_size": 2,
            "extra_constraints": self.sidarthe_extra_1_1_d_2d,
            "test_threshold": 0.1,
            "expected_max_infected": 0.6,
            "test_max_day_threshold": 25,
            "expected_max_day": 47,
        }
        bounds = case["bounds"]()

        scenario = self.make_scenario(
            BilayerDynamics(json_graph=case["model_fn"]()),
            case["initial"](),
            bounds,
            case["identical"](),
            case["steps"],
            # case["query"](case["steps"], case["initial"](), bound=0.33),
            QueryTrue(),
            extra_constraints=case["extra_constraints"](
                case["steps"], case["initial"](), step_size=case["step_size"]
            ),
        )
        config = FUNMANConfig(
            num_steps=case["steps"],
            step_size=case["step_size"],
            solver="dreal",
            initial_state_tolerance=case["initial_state_tolerance"],
        )
        result_sat = Funman().solve(scenario, config=config)
        case["report"](result_sat)

        df = result_sat.dataframe()

        df["infected_states"] = df.apply(
            lambda x: sum([x["I"], x["D"], x["A"], x["R"], x["T"]]), axis=1
        )
        max_infected = df["infected_states"].max()
        max_day = df["infected_states"].idxmax()
        ax = df["infected_states"].plot(
            title=f"I+D+A+R+T by day (max: {max_infected}, day: {max_day})",
        )
        ax.set_xlabel("Day")
        ax.set_ylabel("I+D+A+R+T")
        try:
            plt.savefig(f"s2_1_b_i_infected.png")
        except Exception as e:
            pass
        plt.clf()

        assert (
            abs(max_infected - case["expected_max_infected"])
            < case["test_threshold"]
        )
        assert (
            abs(max_day - case["expected_max_day"])
            < case["test_max_day_threshold"]
        )

        pass

    # @unittest.skip("tmp")
    def test_scenario_2_1_d_2d(self):
        self.iteration = 0
        case = {
            "model_fn": self.sidarthe_bilayer,
            "initial": self.initial_state_sidarthe,
            "bounds": self.bounds_sidarthe,
            "identical": self.sidarthe_identical,
            "steps": 96,
            "query": self.sidarthe_query_1_1_d_1d,
            "report": self.report,
            "initial_state_tolerance": 0,
            "step_size": 12,
            "extra_constraints": self.sidarthe_extra_1_1_d_2d,
        }
        bounds = case["bounds"]()
        bounds["epsilon"] = [0.09, 0.175]
        # epsilon_tolerance = 1e-1
        # theta_tolerance = 0.001
        # bounds["epsilon"] = [
        #     bounds["epsilon"][0],
        #     bounds["epsilon"][1] + epsilon_tolerance,
        # ]
        bounds["theta"] = [
            0.2,
            0.3
            # max(2 * bounds["epsilon"][1], bounds["theta"][1]),
        ]
        scenario = self.make_scenario(
            BilayerDynamics(json_graph=case["model_fn"]()),
            case["initial"](),
            bounds,
            case["identical"](),
            case["steps"],
            # case["query"](),
            case["query"](
                case["steps"],
                case["initial"](),
                bound=0.33,
                step_size=case["step_size"],
            ),
            extra_constraints=case["extra_constraints"](
                case["steps"], case["initial"](), step_size=case["step_size"]
            ),
        )
        config = FUNMANConfig(
            num_steps=case["steps"],
            step_size=case["step_size"],
            solver="dreal",
            initial_state_tolerance=case["initial_state_tolerance"],
        )
        result_sat = Funman().solve(scenario, config=config)
        case["report"](result_sat)

        # case = {
        #     "model_fn": self.sidarthe_bilayer,
        #     "initial": self.initial_state_sidarthe,
        #     "bounds": self.bounds_sidarthe,
        #     "identical": self.sidarthe_identical,
        #     "steps": 100,
        #     "query": self.sidarthe_query_1_1_d_1d,
        #     "report": self.report,
        #     "initial_state_tolerance": 0,
        #     "extra_constraints": self.sidarthe_extra_1_1_d_2d,
        #     "step_size": 10,
        # }
        # bounds = case["bounds"]()
        # epsilon_tolerance = 1e-2
        # theta_tolerance = 0.0005
        # bounds["epsilon"] = [
        #     bounds["epsilon"][0],
        #     bounds["epsilon"][1] + epsilon_tolerance,
        # ]
        # bounds["epsilon"] = [0.16, 0.19]
        # bounds["theta"] = [
        #     2 * bounds["epsilon"][0],
        #     2 * bounds["epsilon"][1],
        # ]
        scenario = self.make_ps_scenario(
            BilayerDynamics(json_graph=case["model_fn"]()),
            case["initial"](),
            bounds,
            case["identical"](),
            case["steps"],
            case["query"](
                case["steps"],
                case["initial"](),
                bound=0.33,
                step_size=case["step_size"],
            ),
            params_to_synth=["epsilon", "theta"],
            extra_constraints=case["extra_constraints"](
                case["steps"], case["initial"](), step_size=case["step_size"]
            ),
        )
        config = FUNMANConfig(
            num_steps=case["steps"],
            solver="dreal",
            number_of_processes=1,
            num_initial_boxes=128,
            initial_state_tolerance=case["initial_state_tolerance"],
            tolerance=1e-4,
            save_smtlib=False,
            step_size=case["step_size"],
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
        with open("funman_s2_1_d_parameter_space_.json", "w") as f:
            f.write(result_sat.parameter_space.json())
        # case["report"](result_sat)


if __name__ == "__main__":
    unittest.main()
