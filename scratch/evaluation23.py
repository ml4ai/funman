import json
import math
import os
import textwrap
import unittest

import matplotlib.pyplot as plt
import pandas as pd
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


class TestUseCases(unittest.TestCase):
    results_df = pd.DataFrame()

    def initial_state(self):
        N0 = 500e3
        E0, I0, R0 = 100, 1, 0
        S0 = N0 - E0 - I0
        init_values = {"S": S0, "E": E0, "I": I0, "R": R0, "D": 0}
        return init_values

    def parameter_bounds(self):
        default_bounds = [1e-3, 0.35]
        bounds = {
            "mu_s": default_bounds,
            "mu_e": default_bounds,
            "mu_i": default_bounds,
            "mu_r": default_bounds,
            "beta_1": default_bounds,
            "epsilon": default_bounds,
            "alpha": default_bounds,
            "gamma": default_bounds,
        }
        return bounds

    def simA_bounds(
        self,
        tolerance=0.0,
        relax=[
            "mu_s",
            "mu_e",
            "mu_i",
            "mu_r",
            "beta_1",
            # "epsilon",
        ],
    ):

        bounds = {
            "mu_s": [0.01, 0.01],
            "mu_e": [0.01, 0.01],
            "mu_i": [0.01, 0.01],
            "mu_r": [0.01, 0.01],
            "beta_1": [0.35, 0.35],
            "epsilon": [0.2, 0.2],
            "alpha": [0.002, 0.002],
            "gamma": [1.0 / 14.0, 1.0 / 14.0],
        }
        bounds = {
            k: (
                [b[0] * (1.0 - tolerance), b[1] * (1.0 + tolerance)]
                if k in relax
                else b
            )
            for k, b in bounds.items()
        }
        return bounds

    def identical_parameters(self):
        identical_parameters = [["mu_s", "mu_e", "mu_i", "mu_r"]]
        return identical_parameters

    def make_global_bounds(self, steps, init_values):
        global_bounds = And(
            [
                And(
                    GE(Symbol(f"{v}_{i}", REAL), Real(-1.0)),
                    LE(
                        Symbol(f"{v}_{i}", REAL),
                        Real(
                            init_values["S"]
                            + init_values["E"]
                            + init_values["I"]
                            + init_values["R"]
                            + init_values["D"]
                        ),
                    ),
                )
                for v in init_values
                for i in range(steps + 1)
            ]
        )
        return global_bounds

    def make_basic_query(self, steps, init_values):

        # Query for test case 1
        query = QueryEncoded()
        query._formula = self.make_global_bounds(steps, init_values)

        return query

    def make_max_difference_constraint(self, steps, init_values, diff=1.0e3):
        # | v_i - v_{i+1} | < diff * v_i, for all v
        constraints = []
        for v in init_values:
            for i in range(steps):
                vi = Symbol(f"{v}_{i}", REAL)
                vj = Symbol(f"{v}_{i+1}", REAL)
                # b = Times(Real(diff), vi)
                # bm = Times(Real(-diff), vi)
                b = Real(diff)
                bm = Real(-diff)
                constraint = And(LE(Minus(vi, vj), b), LE(bm, Minus(vi, vj)))

                constraints.append(constraint)

        return And(constraints)

    def make_monotone_constraints(self, steps, init_values):
        # | v_i - v_{i+1} | < diff * v_i, for all v
        constraints = []
        for (v, dir) in {("S", "decrease"), ("R", "increase")}:
            for i in range(steps):
                vi = Symbol(f"{v}_{i}", REAL)
                vj = Symbol(f"{v}_{i+1}", REAL)

                if dir == "increase":
                    constraint = LT(vi, vj)
                else:
                    constraint = LT(vj, vi)

                constraints.append(constraint)

        return And(constraints)

    def make_peak_constraints(
        self, steps, init_values, window_factor=0.1, tolerance=0.1
    ):
        # peak happens sometime in the middle
        midpoint = math.ceil(steps / 2)
        lb = math.floor(midpoint - (midpoint * window_factor))
        ub = math.ceil(midpoint + (midpoint * window_factor))
        # | v_i - v_{i+1} | < diff * v_i, for all v
        constraints = []
        for v in {"E", "I"}:
            var_constraints = []
            for i in range(lb, ub):
                vi = Symbol(f"{v}_{i}", REAL)
                vj = Symbol(f"{v}_{i+1}", REAL)
                b = Real(tolerance)
                bm = Real(-tolerance)

                constraint = And(LE(Minus(vi, vj), b), LE(bm, Minus(vi, vj)))

                var_constraints.append(constraint)
            constraints.append(Or(var_constraints))

        return And(constraints)

    def make_well_formed_query(self, steps, init_values):

        # Query for test case 1
        query = QueryEncoded()
        query._formula = And(
            [
                # GT(
                #     Symbol(f"R_{steps}", REAL), Real(5.0)
                # ),  # R is near 10 at day 80
                # LT(
                #     Symbol(f"I_{steps}", REAL), Real(1000.0)
                # ),  # I is less than 4.0 on day 30
                # LT(
                #     Symbol(f"S_{steps}", REAL), Real(1000.0)
                # ),  # S is less than 0.5 on day 80
                # LT(
                #     Symbol(f"E_{steps}", REAL), Real(100.0)
                # ),  # S is less than 0.5 on day 80
                # LT(
                #     Symbol(f"D_{steps}", REAL), Real(100.0)
                # ),  # S is less than 0.5 on day 80
                # self.make_global_bounds(steps, init_values),
                # self.make_max_difference_constraint(steps, init_values),
                # self.make_monotone_constraints(steps, init_values),
                # self.make_peak_constraints(steps, init_values),
            ]
        )
        return query

    def make_ps_scenario(
        self,
        bilayer,
        init_values,
        parameter_bounds,
        identical_parameters,
        steps,
        query,
    ):
        model = BilayerModel(
            bilayer=bilayer,
            init_values=init_values,
            parameter_bounds=parameter_bounds,
            identical_parameters=identical_parameters,
        )
        parameters = [
            Parameter(
                name="beta_1",
                lb=parameter_bounds["beta_1"][0],
                ub=parameter_bounds["beta_1"][1],
            ),
            Parameter(
                name="mu_s",
                lb=parameter_bounds["mu_s"][0],
                ub=parameter_bounds["mu_s"][1],
            ),
        ]
        scenario = ParameterSynthesisScenario(
            parameters=parameters, model=model, query=query
        )
        return scenario

    def report(self, result: AnalysisScenario):
        if result.consistent:
            parameters = result._parameters()
            print(f"Iteration {self.iteration}: {parameters}")

            res = pd.Series(name=self.iteration, data=parameters).to_frame().T
            self.results_df = pd.concat([self.results_df, res])
            result.scenario.model.bilayer.to_dot(
                values=result.scenario.model.variables()
            ).render(f"bilayer_{self.iteration}")
            print(result.dataframe())
            result.plot(
                variables=["S", "I", "R"],
                title="\n".join(textwrap.wrap(str(parameters), width=75)),
            )
            plt.savefig(f"bilayer_{self.iteration}.png")
            plt.clf()
        else:
            print(f"Iteration {self.iteration}: is inconsistent")

        self.iteration += 1

    def unit_test_1_bounds(
        self,
        tolerance=0.0,
        N=10.0e6,
        mu=0.00008,
        relax=[
            "mu_s",
            "mu_e",
            "mu_i",
            "mu_r",
            "beta_1",
            # "epsilon",
        ],
    ):

        # R0 = 5.72
        # Lambda = mu*N

        bounds = {
            "mu_s": [mu, mu],
            "mu_e": [mu, mu],
            "mu_i": [mu, mu],
            "mu_r": [mu, mu],
            "beta_1": [0.75, 0.75],
            "epsilon": [0.33, 0.33],
            "alpha": [0.006, 0.006],
            "gamma": [0.125, 0.125],
        }
        bounds = {
            k: (
                [
                    max(b[0] * (1.0 - tolerance), 0.0),
                    min(b[1] * (1.0 + tolerance), 1.0),
                ]
                if k in relax
                else b
            )
            for k, b in bounds.items()
        }
        return bounds

    def unit_test_1_initial_state(self):
        N0 = 10
        E0, I0, R0 = 2e-2, 1e-6, 0
        S0 = N0 - E0 - I0
        init_values = {"S": S0, "E": E0, "I": I0, "R": R0, "D": 0}
        return init_values

    def unit_test_1_query(self, steps, init_values):

        # Query for test case 1
        query = QueryEncoded()
        query._formula = And(
            [
                LT(Symbol(f"S_{steps}", REAL), Real(init_values["S"])),
                GE(Symbol(f"R_{steps}", REAL), Real(init_values["R"]))
                # GT(
                #     Symbol(f"R_{steps}", REAL), Real(5.0)
                # ),  # R is near 10 at day 80
                # LT(
                #     Symbol(f"I_{steps}", REAL), Real(1000.0)
                # ),  # I is less than 4.0 on day 30
                # LT(
                #     Symbol(f"S_{steps}", REAL), Real(1000.0)
                # ),  # S is less than 0.5 on day 80
                # LT(
                #     Symbol(f"E_{steps}", REAL), Real(100.0)
                # ),  # S is less than 0.5 on day 80
                # LT(
                #     Symbol(f"D_{steps}", REAL), Real(100.0)
                # ),  # S is less than 0.5 on day 80
                # self.make_global_bounds(steps, init_values),
                # self.make_max_difference_constraint(steps, init_values),
                # self.make_monotone_constraints(steps, init_values),
                # self.make_peak_constraints(steps, init_values),
            ]
        )

        return query

    def unit_test_1_well_behaved_query(self, steps, init_values):

        # Query for test case 1
        query = QueryEncoded()
        query._formula = And(
            [
                # LT(Symbol(f"S_{steps}", REAL), Real(init_values["S"])),
                # GE(Symbol(f"R_{steps}", REAL), Real(init_values["R"])),
                # GT(
                #     Symbol(f"R_{steps}", REAL), Real(5.0)
                # ),  # R is near 10 at day 80
                # LT(
                #     Symbol(f"I_{steps}", REAL), Real(1000.0)
                # ),  # I is less than 4.0 on day 30
                # LT(
                #     Symbol(f"S_{steps}", REAL), Real(1000.0)
                # ),  # S is less than 0.5 on day 80
                # LT(
                #     Symbol(f"E_{steps}", REAL), Real(100.0)
                # ),  # S is less than 0.5 on day 80
                # LT(
                #     Symbol(f"D_{steps}", REAL), Real(100.0)
                # ),  # S is less than 0.5 on day 80
                self.make_global_bounds(steps, init_values),
                # self.make_max_difference_constraint(steps, init_values),
                # self.make_monotone_constraints(steps, init_values),
                # self.make_peak_constraints(steps, init_values),
            ]
        )

        return query

    def sidarthe_bilayer(self):
        with open(
            os.path.join(RESOURCES, "bilayer", "SIDARTHE_BiLayer.json"), "r"
        ) as f:
            bilayer = json.load(f)
        return bilayer

    def make_scenario(
        self,
        bilayer,
        init_values,
        parameter_bounds,
        identical_parameters,
        steps,
        query,
    ):
        model = BilayerModel(
            bilayer=bilayer,
            init_values=init_values,
            parameter_bounds=parameter_bounds,
            identical_parameters=identical_parameters,
        )

        scenario = ConsistencyScenario(model=model, query=query)
        return scenario

    def sir_bilayer(self):
        with open(
            os.path.join(
                RESOURCES, "bilayer", "CHIME_SIR_dynamics_BiLayer.json"
            ),
            "r",
        ) as f:
            bilayer = json.load(f)

        return bilayer

    def initial_state_sir(self):
        population = 6000
        infected = 3
        return {"S": population - infected, "I": infected, "R": 0}

    def bounds_sir(self):
        R0 = 5.0
        gamma = 1.0 / 14.0
        beta = R0 * gamma
        return {"beta": [beta, beta], "gamma": [gamma, gamma]}

    def sir_query(self):
        return QueryTrue()

    def sir_identical(self):
        return []

    def test_scenario_1_a(self):

        self.iteration = 0

        case_sir = {
            "model_fn": self.sir_bilayer,  # TODO Change to SIR-stratified
            "initial": self.initial_state_sir,
            "bounds": self.bounds_sir,
            "identical": self.sir_identical,
            "steps": 10,
            "query": self.sir_query,
            "report": self.report,
        }
        case_sir_stratified = {}  # TODO
        case = case_sir

        scenario = self.make_scenario(
            BilayerDynamics(json_graph=case["model_fn"]()),
            case["initial"](),
            case["bounds"](),
            case["identical"](),
            case["steps"],
            case["query"](),
        )
        config = FUNMANConfig(max_steps=case["steps"], solver="dreal")
        result_sat = Funman().solve(scenario, config=config)
        case["report"](result_sat)
        pass


if __name__ == "__main__":
    unittest.main()
