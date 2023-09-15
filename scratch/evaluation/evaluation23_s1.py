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
    def sir_bilayer(self):
        with open(
            os.path.join(
                RESOURCES, "bilayer", "CHIME_SIR_dynamics_BiLayer.json"
            ),
            "r",
        ) as f:
            bilayer = json.load(f)

        return bilayer

    def sir_strata_bilayer(self):
        with open(
            os.path.join(RESOURCES, "bilayer", "CHIME_SIR_three_age.json"),
            "r",
        ) as f:
            bilayer = json.load(f)
        return bilayer

    def initial_state_sir(self, population=6000):
        infected = 3
        return {"S": population - infected, "I": infected, "R": 0}

    def initial_state_sir_strat(self):
        population = 2000
        infected = 1
        return {
            "So": population - infected,
            "Io": infected,
            "Ro": 0,
            "Sm": population - infected,
            "Im": infected,
            "Rm": 0,
            "Sy": population - infected,
            "Iy": infected,
            "Ry": 0,
        }

    def bounds_sir(self, population=6000.0):
        R0 = 5.0
        gamma = 1.0 / 14.0
        beta = (R0 * gamma) / population
        return {"beta": [beta, beta], "gamma": [gamma, gamma]}

    def bounds_sir_strat(self, population=2000, noise=0.5):
        R0 = 5.0
        gamma = 1.0 / 14.0
        groups = ["o", "y", "m"]
        inf_matrix = [
            [
                (R0 * gamma) / population,
                (R0 * gamma) / population,
                (R0 * gamma) / population,
            ],
            [
                (R0 * gamma) / population,
                (R0 * gamma) / population,
                (R0 * gamma) / population,
            ],
            [
                (R0 * gamma) / population,
                (R0 * gamma) / population,
                (R0 * gamma) / population,
            ],
        ]
        params = {
            f"inf_{groups[i]}_{groups[j]}": [
                inf_matrix[i][j],
                inf_matrix[i][j],
            ]
            for i in range(len(inf_matrix))
            for j in range(len(inf_matrix[i]))
        }
        params[f"rec_o_o"] = [gamma, gamma]
        params[f"rec_y_y"] = [gamma, gamma]
        params[f"rec_m_m"] = [gamma, gamma]

        params = {
            k: [v[0] * (1.0 - noise), v[1] * (1.0 + noise)]
            for k, v in params.items()
        }

        return params

    def bounds_sir_strat_fixed(self, noise=0.0):
        if_lb = 1e-6
        if_ub = 1e-3  # 1e-1
        rec_lb = 1.0 / 30.0
        rec_ub = 1.0 / 7.0  # 4e0
        params = {
            "inf_y_y": [if_lb, if_ub],
            "inf_y_m": [if_lb, if_ub],
            "inf_y_o": [if_lb, if_ub],
            "inf_m_y": [if_lb, if_ub],
            "inf_m_m": [if_lb, if_ub],
            "inf_m_o": [if_lb, if_ub],
            "inf_o_y": [if_lb, if_ub],
            "inf_o_m": [if_lb, if_ub],
            "inf_o_o": [if_lb, if_ub],
            "rec_y_y": [rec_lb, rec_ub],
            "rec_m_m": [rec_lb, rec_ub],
            "rec_o_o": [rec_lb, rec_ub],
        }
        return params

    def sir_query(self):
        return QueryTrue()

    def sir_strat_query(self, steps, init_values):
        query = QueryEncoded()
        query._formula = And(
            [
                self.make_global_bounds(steps, init_values),
                # GE(Symbol(f"Ro_{steps}", REAL), Symbol(f"Ro_0", REAL)),
                # GE(Symbol(f"Ry_{steps}", REAL), Symbol(f"Ry_0", REAL)),
                # GE(Symbol(f"Rm_{steps}", REAL), Symbol(f"Rm_0", REAL)),
                # LE(Symbol(f"Ro_{steps}", REAL), Real(0.1)),
                # LE(Symbol(f"Ry_{steps}", REAL), Real(0.1)),
                # LE(Symbol(f"Rm_{steps}", REAL), Real(0.1)),
                # GE(Symbol(f"Io_{steps}", REAL), Symbol(f"Io_0", REAL)),
                # GE(Symbol(f"Iy_{steps}", REAL), Symbol(f"Iy_0", REAL)),
                # GE(Symbol(f"Im_{steps}", REAL), Symbol(f"Im_0", REAL)),
                # LE(Symbol(f"Io_{steps}", REAL), Real(2.0)),
                # LE(Symbol(f"Iy_{steps}", REAL), Real(2.0)),
                # LE(Symbol(f"Im_{steps}", REAL), Real(2.0)),
                # LE(Symbol(f"So_{steps}", REAL), Symbol(f"So_0", REAL)),
                # LE(Symbol(f"Sy_{steps}", REAL), Symbol(f"Sy_0", REAL)),
                # LE(Symbol(f"Sm_{steps}", REAL), Symbol(f"Sm_0", REAL)),
                # GE(Symbol(f"So_{steps}", REAL), Real(1997.0)),
                # GE(Symbol(f"Sy_{steps}", REAL), Real(1997.0)),
                # GE(Symbol(f"Sm_{steps}", REAL), Real(1997.0)),
                # self.make_max_difference_constraint(
                #     steps, init_values, diff=1000.0
                # ),
                # self.make_monotone_constraints(steps, init_values),
                TestUseCases.make_monotone_constraints(
                    steps,
                    init_values,
                    {
                        ("So", "decrease"),
                        ("Sm", "decrease"),
                        ("Sy", "decrease"),
                    },
                ),
            ]
        )
        return query
        # return QueryTrue()

    def sir_identical(self):
        return []

    def sir_strat_identical(self):
        return [
            ["rec_o_o", "rec_y_y", "rec_m_m"],
            [
                "inf_o_o",
                "inf_o_y",
                "inf_o_m",
                "inf_y_o",
                "inf_y_y",
                "inf_y_m",
                "inf_m_o",
                "inf_m_y",
                "inf_m_m",
            ],
        ]

    def test_scenario_1_a(self):
        self.iteration = 0

        case_sir = {
            "model_fn": self.sir_bilayer,
            "initial": self.initial_state_sir,
            "bounds": self.bounds_sir,
            "identical": self.sir_identical,
            "steps": 10,
            "query": self.sir_query,
            "report": self.report,
        }
        case_sir_stratified = {
            "model_fn": self.sir_strata_bilayer,
            "initial": self.initial_state_sir_strat,
            # "bounds": self.bounds_sir_strat,
            "bounds": self.bounds_sir_strat_fixed,
            "identical": self.sir_strat_identical,
            "steps": 1,
            "query": self.sir_strat_query,
            "report": self.report,
        }

        case_sir_stratified_ps = {
            "model_fn": self.sir_strata_bilayer,
            "initial": self.initial_state_sir_strat,
            "bounds": self.bounds_sir_strat_fixed,
            "identical": self.sir_strat_identical,
            "steps": 3,
            "query": self.sir_strat_query,
            "report": self.report,
            "noise": 10.0,
            "params_to_synth": ["inf_o_o", "rec_o_o"],
        }

        case = case_sir_stratified
        bounds = case["bounds"](noise=0.0)

        scenario = self.make_scenario(
            BilayerDynamics(json_graph=case["model_fn"]()),
            case["initial"](),
            bounds,
            case["identical"](),
            case["steps"],
            case["query"](case["steps"], case["initial"]()),
        )
        config = FUNMANConfig(num_steps=case["steps"], solver="dreal")
        # result_sat = Funman().solve(scenario, config=config)
        # case["report"](result_sat)

        # Do parameter synth
        case = case_sir_stratified_ps
        bounds = case["bounds"](noise=case["noise"])
        scenario = self.make_ps_scenario(
            BilayerDynamics(json_graph=case["model_fn"]()),
            case["initial"](),
            bounds,
            case["identical"](),
            case["steps"],
            case["query"](case["steps"], case["initial"]()),
            case["params_to_synth"],
        )
        config.tolerance = 1e-4
        config.number_of_processes = 1
        config.num_initial_boxes = 1
        config.num_steps = case["steps"]
        config._handler = ResultCombinedHandler(
            [
                ResultCacheWriter(f"box_search.json"),
                RealtimeResultPlotter(
                    scenario.parameters,
                    plot_points=True,
                    title=f"Feasible Regions (beta)",
                    realtime_save_path=f"box_search.png",
                    dpi=600,
                ),
            ]
        )
        result_sat = Funman().solve(scenario, config=config)


if __name__ == "__main__":
    unittest.main()
