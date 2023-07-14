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
            os.path.join(RESOURCES, "bilayer", "SIDARTHEV_BiLayer.json"),
            "r",
        ) as f:
            bilayer = json.load(f)
        return bilayer

    def initial_state_sidarthe(self):
        with open(
            os.path.join(
                RESOURCES,
                "evaluation23",
                "SIDARTHEV",
                "SIDARTHEV-ic-unit1.json",
            ),
            "r",
        ) as f:
            return json.load(f)

    def bounds_sidarthe(self):
        with open(
            os.path.join(
                RESOURCES,
                "evaluation23",
                "SIDARTHEV",
                "SIDARTHEV-params-unit1.json",
            ),
            "r",
        ) as f:
            params = json.load(f)
        return {k: [v, v] for k, v in params.items()}

    def make_bounds(self, steps, init_values):
        return And(
            [
                Equals(
                    Plus([Symbol(f"{v}_{i}", REAL) for v in init_values]),
                    Real(1.0),
                )
                for i in range(steps + 1)
            ]
        )

    def sidarthe_query(self, steps, init_values):
        query = QueryEncoded()
        query._formula = self.make_bounds(steps, init_values)
        print(query._formula)
        return query

    def sidarthe_identical(self):
        return []

    def test_scenario_2_a(self):
        self.iteration = 0
        case = {
            "model_fn": self.sidarthe_bilayer,
            "initial": self.initial_state_sidarthe,
            "bounds": self.bounds_sidarthe,
            "identical": self.sidarthe_identical,
            "steps": 2,
            "query": self.sidarthe_query,
            "report": self.report,
            "initial_state_tolerance": 1e-6,
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


if __name__ == "__main__":
    unittest.main()
