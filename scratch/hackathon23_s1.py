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


class TestUseCases(unittest.TestCase):
    results_df = pd.DataFrame()

    def initial_bilayer(self):
        bilayer_src1 = {
            "Wa": [
                {"influx": 1, "infusion": 2},  # (beta_1, E')
                {"influx": 2, "infusion": 4},  # (gamma, R')
                {"influx": 8, "infusion": 3},  # (epsilon, I')
                {"influx": 3, "infusion": 5},  # (mu_s, D')
                {"influx": 4, "infusion": 5},  # (mu_e, D')
                {"influx": 5, "infusion": 5},  # (mu_i, D')
                {"influx": 6, "infusion": 5},  # (mu_r, D')
                {"influx": 7, "infusion": 5},  # (alpha, D')
            ],
            "Win": [
                {"arg": 1, "call": 1},  # (S, beta_1)
                {"arg": 1, "call": 3},  # (S, mu_s)
                {"arg": 3, "call": 1},  # (I, beta_1)
                {"arg": 3, "call": 2},  # (I, gamma)
                {"arg": 2, "call": 4},  # (E, mu_e)
                {"arg": 2, "call": 8},  # (E, epsilon)
                {"arg": 3, "call": 7},  # (I, alpha)
                {"arg": 3, "call": 5},  # (I, mu_i)
                {"arg": 4, "call": 6},  # (R, mu_r)
            ],
            "Box": [
                {
                    "parameter": "beta_1",
                    "metadata": {
                        "ref": "http://34.230.33.149:8772/askemo:0000008",
                        "type": "float",
                        "lb": "0.0",
                    },
                },  # 1
                {"parameter": "gamma"},  # 2
                {"parameter": "mu_s"},  # 3
                {"parameter": "mu_e"},  # 4
                {"parameter": "mu_i"},  # 5
                {"parameter": "mu_r"},  # 6
                {"parameter": "alpha"},  # 7
                {"parameter": "epsilon"},  # 8
            ],
            "Qin": [
                {
                    "variable": "S",
                    "metadata": {
                        "ref": "http://34.230.33.149:8772/askemo:0000001"
                    },
                },
                {"variable": "E"},
                {"variable": "I"},
                {"variable": "R"},
                {"variable": "D"},
            ],
            "Qout": [
                {"tanvar": "S'"},
                {"tanvar": "E'"},
                {"tanvar": "I'"},
                {"tanvar": "R'"},
                {"tanvar": "D'"},
            ],
            "Wn": [
                {"efflux": 3, "effusion": 1},  # (mu, S')
                {"efflux": 1, "effusion": 1},  # (beta_1, S')
                {"efflux": 4, "effusion": 2},  # (mu_e, E')
                {"efflux": 8, "effusion": 2},  # (epsilon, E')
                {"efflux": 6, "effusion": 4},  # (mu_r, R')
                {"efflux": 2, "effusion": 3},  # (gamma, I')
                {"efflux": 5, "effusion": 3},  # (mu_i, I')
                {"efflux": 7, "effusion": 3},  # (mu_i, I')
            ],
        }
        return bilayer_src1

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
        for v, dir in {("S", "decrease"), ("R", "increase")}:
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
            ModelParameter(
                name="beta_1",
                lb=parameter_bounds["beta_1"][0],
                ub=parameter_bounds["beta_1"][1],
            ),
            ModelParameter(
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
                variables=["S", "E", "I", "R"],
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

    def test_scenario_1_1_a_unit_test_1(self):
        steps = 3
        self.iteration = 0
        mu = [0.00008, 0.078, 0.19]
        config = FUNMANConfig(num_steps=steps, solver="dreal")

        bilayer = BilayerDynamics(json_graph=self.initial_bilayer())
        bounds = self.unit_test_1_bounds()

        # well_formed_query = self.make_well_formed_query(
        #     steps, self.initial_state()
        # )

        ###########################################################
        # Unit Test 1, using Paper Parameters
        ###########################################################
        testcase = 0
        # print("Dynamics + simA params + well formed ...")
        print(f"Bounds: {self.unit_test_1_bounds()}")
        scenario = self.make_scenario(
            bilayer,
            self.initial_state(),
            self.unit_test_1_bounds(mu=mu[testcase]),
            self.identical_parameters(),
            steps,
            self.unit_test_1_query(steps, self.initial_state()),
        )
        result_sat = Funman().solve(scenario, config=config)
        self.report(result_sat)

        ###########################################################
        # The basic constraints are satisfied, but output diverges,
        # need to check that population is consistent (doesn't exceed N)
        # Generate results using any parameters
        ###########################################################
        print(f"Bounds: {self.unit_test_1_bounds(mu=mu[testcase])}")
        bounds = self.unit_test_1_bounds(mu=mu[testcase])
        # bounds["beta_1"] = [0.0001, 0.0001]
        scenario = self.make_scenario(
            bilayer,
            self.initial_state(),
            bounds,
            self.identical_parameters(),
            steps,
            self.unit_test_1_well_behaved_query(steps, self.initial_state()),
        )
        result_sat = Funman().solve(scenario, config=config)
        self.report(result_sat)

        ###########################################################
        # Basic constraints are not satisfied, so relax them
        ###########################################################
        bounds = self.unit_test_1_bounds(
            mu=mu[testcase],
        )
        bounds["beta_1"] = [1e-6, 2e-1]
        scenario = self.make_scenario(
            bilayer,
            self.initial_state(),
            bounds,
            self.identical_parameters(),
            steps,
            self.unit_test_1_well_behaved_query(steps, self.initial_state()),
        )
        result_sat = Funman().solve(scenario, config=config)
        self.report(result_sat)

        ###########################################################
        # Unit Test 2, using Paper Parameters
        ###########################################################
        testcase = 1
        # print("Dynamics + simA params + well formed ...")
        print(f"Bounds: {bounds}")
        scenario = self.make_scenario(
            bilayer,
            self.initial_state(),
            self.unit_test_1_bounds(mu=mu[testcase]),
            self.identical_parameters(),
            steps,
            self.unit_test_1_query(steps, self.initial_state()),
        )
        result_sat = Funman().solve(scenario, config=config)
        self.report(result_sat)

        ###########################################################
        # The basic constraints are satisfied, but output diverges,
        # need to check that population is consistent (doesn't exceed N)
        # Generate results using any parameters
        ###########################################################
        print(f"Bounds: {bounds}")
        scenario = self.make_scenario(
            bilayer,
            self.initial_state(),
            self.unit_test_1_bounds(mu=mu[testcase]),
            self.identical_parameters(),
            steps,
            self.unit_test_1_well_behaved_query(steps, self.initial_state()),
        )
        result_sat = Funman().solve(scenario, config=config)
        self.report(result_sat)

        ###########################################################
        # Basic constraints are not satisfied, so relax them
        ###########################################################
        scenario = self.make_scenario(
            bilayer,
            self.initial_state(),
            self.unit_test_1_bounds(
                mu=mu[testcase],
                tolerance=3.0,
                relax=[
                    # "mu_s",
                    # "mu_e",
                    # "mu_i",
                    # "mu_r",
                    # "beta_1",
                    # "epsilon",
                    "alpha",
                    "gamma",
                ],
            ),
            self.identical_parameters(),
            steps,
            self.unit_test_1_well_behaved_query(steps, self.initial_state()),
        )
        result_sat = Funman().solve(scenario, config=config)
        self.report(result_sat)

        ###########################################################
        # Unit Test 3, using Paper Parameters
        ###########################################################
        testcase = 2
        # print("Dynamics + simA params + well formed ...")
        print(f"Bounds: {bounds}")
        scenario = self.make_scenario(
            bilayer,
            self.initial_state(),
            self.unit_test_1_bounds(mu=mu[testcase]),
            self.identical_parameters(),
            steps,
            self.unit_test_1_query(steps, self.initial_state()),
        )
        result_sat = Funman().solve(scenario, config=config)
        self.report(result_sat)

        ###########################################################
        # The basic constraints are satisfied, but output diverges,
        # need to check that population is consistent (doesn't exceed N)
        # Generate results using any parameters
        ###########################################################
        print(f"Bounds: {bounds}")
        scenario = self.make_scenario(
            bilayer,
            self.initial_state(),
            self.unit_test_1_bounds(mu=mu[testcase]),
            self.identical_parameters(),
            steps,
            self.unit_test_1_well_behaved_query(steps, self.initial_state()),
        )
        result_sat = Funman().solve(scenario, config=config)
        self.report(result_sat)

        ###########################################################
        # Basic constraints are not satisfied, so relax them
        ###########################################################
        scenario = self.make_scenario(
            bilayer,
            self.initial_state(),
            self.unit_test_1_bounds(
                mu=mu[testcase],
                tolerance=3.0,
                relax=[
                    # "mu_s",
                    # "mu_e",
                    # "mu_i",
                    # "mu_r",
                    # "beta_1",
                    # "epsilon",
                    "alpha",
                    "gamma",
                ],
            ),
            self.identical_parameters(),
            steps,
            self.unit_test_1_well_behaved_query(steps, self.initial_state()),
        )
        result_sat = Funman().solve(scenario, config=config)
        self.report(result_sat)

        # print(self.results_df)
        # self.results_df.boxplot().get_figure().savefig("stats.png")

        ###########################################################
        # Synthesize beta_1
        ###########################################################
        bounds = self.unit_test_1_bounds()
        bounds["beta_1"] = [1e-8, 3e-3]
        bounds["mu_s"] = [0.00008, 0.19]
        bounds["mu_i"] = [0.00008, 0.19]
        bounds["mu_e"] = [0.00008, 0.19]
        bounds["mu_r"] = [0.00008, 0.19]
        scenario = self.make_ps_scenario(
            bilayer,
            self.initial_state(),
            bounds,
            self.identical_parameters(),
            steps,
            self.unit_test_1_well_behaved_query(steps, self.initial_state()),
        )
        config.tolerance = 1e-3
        config.number_of_processes = 1
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
        # self.report(result_sat)


if __name__ == "__main__":
    unittest.main()
