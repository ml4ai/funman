import os
import unittest

import matplotlib.pyplot as plt

from funman import Funman
from funman.funman import FUNMANConfig

# from funman.funman import FUNMANConfig
from funman.model import QueryLE
from funman.model.bilayer import BilayerDynamics, BilayerModel
from funman.model.query import QueryTrue
from funman.scenario import ConsistencyScenario, ConsistencyScenarioResult

# from funman_demo.handlers import RealtimeResultPlotter, ResultCacheWriter


# from funman.utils.handlers import ResultCombinedHandler

RESOURCES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../resources"
)


class TestUseCases(unittest.TestCase):
    def setup_use_case_bilayer_common(self):

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
                {"parameter": "beta_1"},  # 1
                {"parameter": "gamma"},  # 2
                {"parameter": "mu_s"},  # 3
                {"parameter": "mu_e"},  # 4
                {"parameter": "mu_i"},  # 5
                {"parameter": "mu_r"},  # 6
                {"parameter": "alpha"},  # 7
                {"parameter": "epsilon"},  # 8
            ],
            "Qin": [
                {"variable": "S"},
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

        init_values = {"S": 1000, "E": 1, "I": 1, "R": 1, "D": 1003}

        mu = [0, 1]
        paramater_bounds = {
            "mu_s": mu,
            "mu_e": mu,
            "mu_i": mu,
            "mu_r": mu,
            "beta_1": [0, 1],
            "epsilon": [0, 1],
            "alpha": [0, 1],
            "gamma": [0, 1],
        }

        bilayer1 = BilayerDynamics(json_graph=bilayer_src1)
        bilayer1.to_dot().render()

        model = BilayerModel(
            bilayer=bilayer1,
            init_values=init_values,
            parameter_bounds=paramater_bounds,
            identical_parameters=[["mu_s", "mu_e", "mu_i", "mu_r"]],
        )

        query = QueryTrue()

        return model, query

    def setup_use_case_bilayer_consistency(self):
        model, query = self.setup_use_case_bilayer_common()

        scenario = ConsistencyScenario(model=model, query=query)
        return scenario

    def test_use_case_bilayer_consistency(self):
        scenario = self.setup_use_case_bilayer_consistency()

        funman = Funman()

        # Show that region in parameter space is sat (i.e., there exists a true point)
        result_sat: ConsistencyScenarioResult = funman.solve(
            scenario, config=FUNMANConfig(max_steps=10)
        )
        df = result_sat.dataframe()
        print(df)

        result_sat.plot()
        plt.savefig("bilayer.png")


if __name__ == "__main__":
    unittest.main()
