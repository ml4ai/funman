import asyncio
import json
import textwrap
import unittest
from os import path

import pandas as pd

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

import matplotlib.pyplot as plt

# From openapi-python-client
from funman_api_client import Client
from funman_api_client.api.default import (
    solve_consistency_solve_consistency_put,
)
from funman_api_client.models import (
    BilayerModel,
    BodySolveConsistencySolveConsistencyPut,
    ConsistencyScenario,
    ConsistencyScenarioResult,
    QueryTrue,
)

API_BASE_PATH = path.join(path.dirname(path.abspath(__file__)), "..")
RESOURCES = path.join(path.dirname(path.abspath(__file__)), "../resources")
API_SERVER_HOST = "0.0.0.0"
API_SERVER_PORT = 8190
SERVER_URL = f"http://{API_SERVER_HOST}:{API_SERVER_PORT}"
OPENAPI_URL = f"{SERVER_URL}/openapi.json"
CLIENT_NAME = "funman-api-client"


class TestAPI(unittest.TestCase):
    results_df = pd.DataFrame()

    def report(self, result):
        if result.consistent:
            parameters = result.consistent.to_dict()
            print(f"Iteration {self.iteration}: {parameters}")

            res = pd.Series(name=self.iteration, data=parameters).to_frame().T
            self.results_df = pd.concat([self.results_df, res])

            df = pd.DataFrame.from_dict(result.timeseries.to_dict())
            df = df.interpolate(method="linear")
            print(df)

            variables = (["S", "E", "I", "R"],)
            plt.show(block=False)
            df[variables].plot(
                marker="o",
                title="\n".join(textwrap.wrap(str(parameters), width=75)),
            )
            plt.savefig(f"bilayer_{self.iteration}.png")
            plt.clf()
        else:
            print(f"Iteration {self.iteration}: is inconsistent")

        self.iteration += 1

    def test_api_consistency(self):
        self.iteration = 0

        funman_client = Client(SERVER_URL, timeout=None)

        # bilayer_path = path.join(RESOURCES, "bilayer", "CHIME_SIR_dynamics_BiLayer.json")
        # with open(bilayer_path, "r") as bl:
        #     bilayer_json = json.load(bl)
        bilayer_json = self.initial_bilayer()

        init_values = self.initial_state()
        parameter_bounds = self.bounds()
        identical_parameters = self.identical_parameters()

        steps = 10

        response = asyncio.run(
            solve_consistency_solve_consistency_put.asyncio_detailed(
                client=funman_client,
                json_body=BodySolveConsistencySolveConsistencyPut(
                    ConsistencyScenario(
                        BilayerModel.from_dict(
                            {
                                "bilayer": {"json_graph": bilayer_json},
                                "init_values": init_values,
                                "parameter_bounds": parameter_bounds,
                                "identical_parameters": identical_parameters,
                            }
                        ),
                        QueryTrue(),
                    )
                ),
            )
        )

        #  ConsistencyScenario.from_dict({
        #      "model": {
        #          "bilayer": {"json_graph": bilayer_json},
        #          "init_values": init_values,
        #          "parameter_bounds": parameter_bounds,
        #          "identical_parameters": identical_parameters,
        #      },
        #      "query": {
        #      },
        #  })

        result = ConsistencyScenarioResult.from_dict(
            src_dict=json.loads(response.content.decode())
        )
        self.report(result)

    def initial_state(self):
        N0 = 500e3
        E0, I0, R0 = 100, 1, 0
        S0 = N0 - E0 - I0
        init_values = {"S": S0, "E": E0, "I": I0, "R": R0, "D": 0}
        return init_values

    def bounds(self):
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

    def identical_parameters(self):
        identical_parameters = [["mu_s", "mu_e", "mu_i", "mu_r"]]
        return identical_parameters

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


if __name__ == "__main__":
    unittest.main()
