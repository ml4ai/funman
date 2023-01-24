import os
import textwrap
import unittest

import matplotlib.pyplot as plt
from pysmt.shortcuts import GE, GT, LE, LT, REAL, And, Equals, Real, Symbol

from funman import Funman
from funman.funman import FUNMANConfig

# from funman.funman import FUNMANConfig
from funman.model import QueryLE
from funman.model.bilayer import BilayerDynamics, BilayerModel
from funman.model.query import QueryEncoded, QueryTrue
from funman.scenario import ConsistencyScenario, ConsistencyScenarioResult

# from funman_demo.handlers import RealtimeResultPlotter, ResultCacheWriter


# from funman.utils.handlers import ResultCombinedHandler

RESOURCES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../resources"
)


class TestUseCases(unittest.TestCase):
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
        init_values = {"S": 10, "E": 0, "I": 0, "R": 0, "D": 10}
        return init_values

    def paramater_bounds(self):
        default_bounds = [0, 1]
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

    def identical_parameters(self):
        identical_parameters = [["mu_s", "mu_e", "mu_i", "mu_r"]]
        return identical_parameters

    def make_query(self):
        # Query for test case 1
        query = QueryEncoded(
            formula=And(
                [
                    GT(
                        Symbol("R_80", REAL), Real(9.5)
                    ),  # R is near 10 at day 80
                    LT(
                        Symbol("I_30", REAL), Real(4.0)
                    ),  # I is less than 4.0 on day 30
                    LT(
                        Symbol("S_80", REAL), Real(0.5)
                    ),  # S is less than 0.5 on day 80
                ]
            )
        )
        return query

    def make_scenario(
        self, bilayer, init_values, parameter_bounds, identical_parameters
    ):
        model = BilayerModel(
            bilayer=bilayer,
            init_values=init_values,
            parameter_bounds=parameter_bounds,
            identical_parameters=identical_parameters,
        )

        query = self.make_query()

        scenario = ConsistencyScenario(model=model, query=query)
        return scenario

    def report(self, bilayer, result):
        parameters = result._parameters()
        print(f"Iteration {self.iteration}: {parameters}")
        bilayer.to_dot().render(f"bilayer_{self.iteration}")
        # print(result.dataframe())
        result.plot(
            variables=["S", "E", "I", "R"],
            title="\n".join(textwrap.wrap(str(parameters), width=60)),
        )
        plt.savefig(f"bilayer_{self.iteration}.png")

        self.iteration += 1

    def test_use_case_bilayer_consistency(self):
        self.iteration = 0
        bilayer = BilayerDynamics(json_graph=self.initial_bilayer())
        bounds = self.paramater_bounds()
        scenario = self.make_scenario(
            bilayer,
            self.initial_state(),
            bounds,
            self.identical_parameters(),
        )

        # TODO: Illustrate bilayer mismatch with code version A
        result_sat = Funman().solve(
            scenario, config=FUNMANConfig(max_steps=80)
        )
        self.report(bilayer, result_sat)

        # Shrink bounds on beta_1 from 0.5 in [0, 1) to [0, 0.1)
        bounds["beta_1"] = [0, 0.1]
        scenario = self.make_scenario(
            bilayer,
            self.initial_state(),
            bounds,
            self.identical_parameters(),
        )
        result_sat = Funman().solve(
            scenario, config=FUNMANConfig(max_steps=80)
        )
        self.report(bilayer, result_sat)


if __name__ == "__main__":
    unittest.main()
