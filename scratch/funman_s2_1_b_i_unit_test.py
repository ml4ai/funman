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
            # os.path.join(RESOURCES, "bilayer", "SIDARTHE_BiLayer.json"),
            os.path.join(
                RESOURCES, "bilayer", "SIDARTHE_BiLayer_corrected.json"
            ),
            "r",
        ) as f:
            bilayer = json.load(f)
        return bilayer

    def sidarthe_bilayer_UF(self):
        with open(
            os.path.join(
                RESOURCES, "bilayer", "SIDARTHE_petri_UF_bilayer.json"
            ),
            "r",
        ) as f:
            bilayer = json.load(f)
        return bilayer

    def sidarthe_bilayer_CM(self):
        with open(
            os.path.join(
                RESOURCES, "bilayer", "SIDARTHE_BiLayer-CTM-correction.json"
            ),
            "r",
        ) as f:
            bilayer = json.load(f)
        return bilayer

    def sidarthe_bilayer_extract(self):
        with open(
            os.path.join(
                RESOURCES,
                "bilayer",
                "SKEMA_SIDARTHE_PN_BL_renamed_transitions.json",
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

    def sidarthe_extra_1_1_d_2d(self, steps, init_values, step_size=1):
        return And(
            [
                self.make_bounds(steps, init_values, step_size=step_size),
            ]
        )

    def sidarthe_identical(self):
        return []

    # @unittest.skip("tmp")
    def test_scenario_2_1_b_i(self):
        self.iteration = 0
        case = {
            "model_fn": self.sidarthe_bilayer,
            "initial": self.initial_state_sidarthe,
            "bounds": self.bounds_sidarthe,
            "identical": self.sidarthe_identical,
            "steps": 100,
            "query": QueryTrue,
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
            case["query"](),
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
            plt.savefig(f"funman_s2_1_b_i_infected.png")
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

    def test_scenario_2_1_b_i_extract(self):
        self.iteration = 0
        case = {
            "name": "funman_s2_1_b_i_SKEMA_extract",
            "model_fn": self.sidarthe_bilayer_extract,
            "initial": self.initial_state_sidarthe,
            "bounds": self.bounds_sidarthe,
            "identical": self.sidarthe_identical,
            "steps": 100,
            "query": QueryTrue,
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
            case["query"](),
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
        case["report"](result_sat, name=case["name"])

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
            plt.savefig(f"{case['name']}_infected.png")
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

    @unittest.skip("tmp")
    def test_scenario_2_1_b_i_UF_DM(self):
        self.iteration = 1

        # Check if existing parameters work

        case = {
            "model_fn": self.sidarthe_bilayer_UF,
            "initial": self.initial_state_sidarthe,
            "bounds": self.bounds_sidarthe,
            "identical": self.sidarthe_identical,
            "steps": 100,
            "query": QueryTrue,
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
        # factor = 1.0
        # bounds = {k: [v[0]*(1.0-factor), v[1]*(1.0+factor)] for k, v in bounds.items()}

        scenario = self.make_scenario(
            BilayerDynamics(json_graph=case["model_fn"]()),
            case["initial"](),
            bounds,
            case["identical"](),
            case["steps"],
            case["query"](),
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

        assert result_sat.consistent

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
            plt.savefig(f"funman_s2_1_b_i_infected_UF.png")
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

    @unittest.skip("tmp")
    def test_scenario_2_1_b_i_UF_DM_Relax(self):
        self.iteration = 2

        # Check if relaxing parameters works

        case = {
            "model_fn": self.sidarthe_bilayer_UF,
            "initial": self.initial_state_sidarthe,
            "bounds": self.bounds_sidarthe,
            "identical": self.sidarthe_identical,
            "steps": 5,
            "query": QueryTrue,
            "report": self.report,
            "initial_state_tolerance": 0,
            "step_size": 1,
            "extra_constraints": self.sidarthe_extra_1_1_d_2d,
            "test_threshold": 0.1,
            "expected_max_infected": 0.6,
            "test_max_day_threshold": 25,
            "expected_max_day": 47,
        }
        bounds = case["bounds"]()
        factor = 1.0
        bounds = {
            k: [v[0] * (1.0 - factor), v[1] * (1.0 + factor)]
            for k, v in bounds.items()
        }

        scenario = self.make_scenario(
            BilayerDynamics(json_graph=case["model_fn"]()),
            case["initial"](),
            bounds,
            case["identical"](),
            case["steps"],
            case["query"](),
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

        assert result_sat.consistent

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
            plt.savefig(f"funman_s2_1_b_i_infected_UF.png")
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
    def test_scenario_2_1_b_i_CMM(self):
        # Manually encoded Bilayer by CM and DM
        self.iteration = 3
        case = {
            "name": "funman_s2_1_b_i_CMM_hand",
            "model_fn": self.sidarthe_bilayer_CM,
            "initial": self.initial_state_sidarthe,
            "bounds": self.bounds_sidarthe,
            "identical": self.sidarthe_identical,
            "steps": 100,
            "query": QueryTrue,
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
            case["query"](),
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
        case["report"](result_sat, name=case["name"])

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
            plt.savefig(f"{case['name']}_infected.png")
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


if __name__ == "__main__":
    unittest.main()
