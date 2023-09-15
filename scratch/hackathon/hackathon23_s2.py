import math
import os
import textwrap
import unittest

import matplotlib.pyplot as plt
import pandas as pd
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
from funman.scenario import ConsistencyScenario, ConsistencyScenarioResult
from funman.scenario.scenario import AnalysisScenario

# from funman_demo.handlers import RealtimeResultPlotter, ResultCacheWriter


# from funman.utils.handlers import ResultCombinedHandler

RESOURCES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../resources"
)


class TestUseCases(unittest.TestCase):
    results_df = pd.DataFrame()

    def initial_bilayer(self):
        bilayer_src1 = {
            "Qin": [
                {"variable": "S"},
                {"variable": "V"},
                {"variable": "I"},
                {"variable": "I_v"},
                {"variable": "R"},
            ],
            "Box": [
                {"parameter": "beta_1"},
                {"parameter": "beta_2"},
                {"parameter": "v_r"},
                {"parameter": "v_s1"},
                {"parameter": "v_s2"},
                {"parameter": "gamma_1"},
                {"parameter": "gamma_2"},
            ],
            "Qout": [
                {"tanvar": "S'"},
                {"tanvar": "V'"},
                {"tanvar": "I'"},
                {"tanvar": "I_v'"},
                {"tanvar": "R'"},
            ],
            "Win": [
                {"arg": 1, "call": 1},
                {"arg": 1, "call": 2},
                {"arg": 1, "call": 3},
                {"arg": 2, "call": 4},
                {"arg": 2, "call": 5},
                {"arg": 3, "call": 1},
                {"arg": 3, "call": 4},
                {"arg": 3, "call": 6},
                {"arg": 4, "call": 2},
                {"arg": 4, "call": 5},
                {"arg": 4, "call": 7},
            ],
            "Wa": [
                {"influx": 1, "infusion": 3},
                {"influx": 2, "infusion": 3},
                {"influx": 3, "infusion": 2},
                {"influx": 4, "infusion": 4},
                {"influx": 5, "infusion": 4},
                {"influx": 6, "infusion": 5},
                {"influx": 7, "infusion": 5},
            ],
            "Wn": [
                {"efflux": 1, "effusion": 1},
                {"efflux": 2, "effusion": 1},
                {"efflux": 3, "effusion": 1},
                {"efflux": 4, "effusion": 2},
                {"efflux": 5, "effusion": 2},
                {"efflux": 6, "effusion": 3},
                {"efflux": 7, "effusion": 4},
            ],
        }
        return bilayer_src1

    def initial_state(self):
        N0 = 10e7
        V0, I0, I_v0, R0 = 7e7, 1e7, 2e6, 0
        S0 = N0 - V0 - I0 - I_v0 - R0
        init_values = {"S": S0, "V": V0, "I": I0, "I_v": I_v0, "R": R0}
        return init_values

    def parameter_bounds(self):
        default_bounds = [1e-3, 0.35]
        bounds = {
            "beta_1": default_bounds,
            "beta_2": default_bounds,
            "gamma_1": default_bounds,
            "gamma_2": default_bounds,
            "v_r": default_bounds,
            "v_s1": default_bounds,
            "v_s2": default_bounds,
        }
        return bounds

    def simA_bounds(
        self,
        tolerance=0.0,
        relax=[
            "beta_1",
            "beta_2",
            "gamma_1",
            "gamma_2",
            "v_r",
            "v_s1",
            "v_s2",
        ],
    ):
        bounds = {
            "beta_1": [0.35, 0.35],
            "beta_2": [0.35, 0.35],
            "gamma_1": [1.0 / 14.0, 1.0 / 14.0],
            "gamma_2": [1.0 / 14.0, 1.0 / 14.0],
            "v_s1": [1.0 / 14.0, 1.0 / 14.0],
            "v_s2": [1.0 / 14.0, 1.0 / 14.0],
            "v_r": [0.75, 0.75],
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
        identical_parameters = [
            ["beta_1", "beta_2"],
            ["gamma_1", "gamma_2"],
            ["v_s1", "v_s2"],
        ]
        return identical_parameters

    def make_global_bounds(self, steps, init_values):
        global_bounds = And(
            [
                And(
                    GE(Symbol(f"{v}_{i}", REAL), Real(0.0)),
                    LE(
                        Symbol(f"{v}_{i}", REAL),
                        Real(
                            init_values["S"]
                            + init_values["V"]
                            + init_values["I"]
                            + init_values["I_v"]
                            + init_values["R"]
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
        for v in {"I_v", "I"}:
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
                variables=["S", "V", "I", "I_v", "R"],
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
        N=10.0e7,
        relax=[
            "beta_1",
            "beta_2",
            "v_r",
            "v_s1",
            "v_s2",
            "gamma_1",
            "gamma_2",
        ],
    ):
        bounds = {
            "beta_1": [0.35, 0.35],
            "beta_2": [0.35, 0.35],
            "gamma_1": [1.0 / 14.0, 1.0 / 14.0],
            "gamma_2": [1.0 / 14.0, 1.0 / 14.0],
            "v_s1": [1.0 / 14.0, 1.0 / 14.0],
            "v_s2": [1.0 / 14.0, 1.0 / 14.0],
            "v_r": [0.75, 0.75],
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
        N0 = 10e7
        V0, I0, I_v0, R0 = 7e7, 1e7, 2e6, 0
        S0 = N0 - V0 - I0 - I_v0 - R0
        init_values = {"S": S0, "V": V0, "I": I0, "I_v": I_v0, "R": R0}
        return init_values

    def unit_test_1_query(self, steps, init_values):
        # Query for test case 1
        query = QueryEncoded()
        query._formula = And(
            [
                LT(Symbol(f"S_{steps}", REAL), Real(init_values["S"])),
                GE(Symbol(f"R_{steps}", REAL), Real(init_values["R"])),
                # LT(Symbol(f"I_{steps}", REAL), Real(20000.0)),
                # LT(Symbol(f"I_v_{steps}", REAL), Real(10000.0)),
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
                LT(Symbol(f"S_{steps}", REAL), Real(init_values["S"])),
                GE(Symbol(f"R_{steps}", REAL), Real(init_values["R"])),
                self.make_global_bounds(steps, init_values),
            ]
        )

        return query

    def test_scenario_1_1_a_unit_test_1(self):
        steps = 3
        self.iteration = 0
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
        # print(f"Bounds: {self.unit_test_1_bounds()}")
        scenario = self.make_scenario(
            bilayer,
            self.initial_state(),
            self.unit_test_1_bounds(),
            self.identical_parameters(),
            steps,
            QueryTrue(),
        )
        result_sat = Funman().solve(scenario, config=config)
        self.report(result_sat)

        ###########################################################
        # The basic constraints are satisfied, but output diverges,
        # need to check that population is consistent (doesn't exceed N)
        # Generate results using any parameters
        ###########################################################
        # print(f"Bounds: {self.unit_test_1_bounds()}")
        scenario = self.make_scenario(
            bilayer,
            self.initial_state(),
            self.unit_test_1_bounds(),
            self.identical_parameters(),
            steps,
            self.unit_test_1_well_behaved_query(steps, self.initial_state()),
        )
        result_sat = Funman().solve(scenario, config=config)
        self.report(result_sat)

        ###########################################################
        # Basic constraints are not satisfied, so relax them
        ###########################################################
        bounds = self.unit_test_1_bounds()
        bounds["beta_1"] = [1e-9, 1e-8]
        bounds["beta_2"] = [1e-9, 1e-8]
        bounds["v_r"] = [1e-5, 1e-4]
        bounds["v_s1"] = [1e-8, 1e-7]
        bounds["v_s2"] = [1e-8, 1e-7]
        bounds["gamma_1"] = [1e-2, 1e-1]
        bounds["gamma_2"] = [1e-2, 1e-1]
        scenario = self.make_scenario(
            bilayer,
            self.initial_state(),
            bounds,
            self.identical_parameters(),
            steps,
            self.unit_test_1_query(steps, self.initial_state()),
            # QueryTrue(),
        )
        result_sat = Funman().solve(scenario, config=config)
        print(result_sat._parameters())
        self.report(result_sat)

        print(self.results_df)
        # self.results_df.boxplot().get_figure().savefig("stats.png")


if __name__ == "__main__":
    unittest.main()
