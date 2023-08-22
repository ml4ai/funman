import asyncio
import json
import unittest
from os import path

import funman.api.client as client
from funman.api.api import app
from funman.api.server import Server, ServerConfig

API_BASE_PATH = path.join(path.dirname(path.abspath(__file__)), "..")
RESOURCES = path.join(path.dirname(path.abspath(__file__)), "../resources")
API_SERVER_HOST = "0.0.0.0"
API_SERVER_PORT = 8190
SERVER_URL = f"http://{API_SERVER_HOST}:{API_SERVER_PORT}"
OPENAPI_URL = f"{SERVER_URL}/openapi.json"
CLIENT_NAME = "funman-api-client"


class TestAPI(unittest.TestCase):
    def test_api_consistency(self):
        # Start API Server

        server = Server(
            config=ServerConfig(
                app,
                host=API_SERVER_HOST,
                port=API_SERVER_PORT,
                log_level="info",
            )
        )
        with server.run_in_thread():
            # Server is started.

            client.make_client(
                API_BASE_PATH, openapi_url=OPENAPI_URL, client_name=CLIENT_NAME
            )
            from funman_api_client import Client
            from funman_api_client.api.default import (
                solve_consistency_solve_consistency_put,
            )
            from funman_api_client.models import (
                BodySolveConsistencySolveConsistencyPut,
                ConsistencyScenario,
                ConsistencyScenarioResult,
            )

            funman_client = Client(SERVER_URL, timeout=None)

            bilayer_path = path.join(
                RESOURCES, "bilayer", "CHIME_SIR_dynamics_BiLayer.json"
            )
            with open(bilayer_path, "r") as bl:
                bilayer_json = json.load(bl)
            infected_threshold = 130
            init_values = {"S": 9998, "I": 1, "R": 1}

            response = asyncio.run(
                solve_consistency_solve_consistency_put.asyncio_detailed(
                    client=funman_client,
                    json_body=BodySolveConsistencySolveConsistencyPut(
                        ConsistencyScenario.from_dict(
                            {
                                "model": {
                                    "init_values": init_values,
                                    "bilayer": {"json_graph": bilayer_json},
                                },
                                "query": {
                                    "variable": "I",
                                    "ub": infected_threshold,
                                    "at_end": False,
                                },
                            }
                        )
                    ),
                )
            )
            result = ConsistencyScenarioResult.from_dict(
                src_dict=json.loads(response.content.decode())
            )
            assert result

    def test_api_parameter_synthesis(self):
        # Start API Server
        server = Server(
            config=ServerConfig(
                app,
                host=API_SERVER_HOST,
                port=API_SERVER_PORT,
                log_level="info",
            )
        )
        with server.run_in_thread():
            # Server is started.

            client.make_client(
                API_BASE_PATH, openapi_url=OPENAPI_URL, client_name=CLIENT_NAME
            )

            from funman_api_client import Client
            from funman_api_client.api.default import (
                solve_parameter_synthesis_solve_parameter_synthesis_put,
            )
            from funman_api_client.models import (
                BodySolveParameterSynthesisSolveParameterSynthesisPut,
                FUNMANConfig,
                ParameterSynthesisScenario,
                ParameterSynthesisScenarioResult,
            )

            funman_client = Client(SERVER_URL, timeout=None)

            bilayer_path = path.join(
                RESOURCES, "bilayer", "CHIME_SIR_dynamics_BiLayer.json"
            )
            with open(bilayer_path, "r") as bl:
                bilayer_json = json.load(bl)

            infected_threshold = 3
            init_values = {"S": 9998, "I": 1, "R": 1}

            lb = 0.000067 * (1 - 0.5)
            ub = 0.000067 * (1 + 0.5)

            response = asyncio.run(
                solve_parameter_synthesis_solve_parameter_synthesis_put.asyncio_detailed(
                    client=funman_client,
                    json_body=BodySolveParameterSynthesisSolveParameterSynthesisPut(
                        ParameterSynthesisScenario.from_dict(
                            {
                                "parameters": [
                                    {
                                        "name": "beta",
                                        "lb": lb,
                                        "ub": ub,
                                    }
                                ],
                                "model": {
                                    "init_values": init_values,
                                    "bilayer": {"json_graph": bilayer_json},
                                    "parameter_bounds": {
                                        "beta": [lb, ub],
                                        "gamma": [1.0 / 14.0, 1.0 / 14.0],
                                    },
                                },
                                "query": {
                                    "variable": "I",
                                    "ub": infected_threshold,
                                    "at_end": False,
                                },
                            }
                        ),
                        FUNMANConfig.from_dict(
                            {"tolerance": 1.0e-8, "number_of_processes": 1}
                        ),
                    ),
                )
            )
            result = ParameterSynthesisScenarioResult.from_dict(
                src_dict=json.loads(response.content.decode())
            )
            assert len(result.parameter_space.true_boxes) > 0
            assert len(result.parameter_space.false_boxes) > 0

    def test_api_simulation(self):
        # Start API Server
        server = Server(
            config=ServerConfig(
                app,
                host=API_SERVER_HOST,
                port=API_SERVER_PORT,
                log_level="info",
            )
        )
        with server.run_in_thread():
            # Server is started.

            client.make_client(
                API_BASE_PATH, openapi_url=OPENAPI_URL, client_name=CLIENT_NAME
            )

            from funman_api_client import Client
            from funman_api_client.api.default import (
                solve_simulation_solve_simulation_put,
            )
            from funman_api_client.models import (
                AnalysisScenarioResultException,
                BodySolveSimulationSolveSimulationPut,
                FUNMANConfig,
                SimulationScenario,
                SimulationScenarioResult,
            )

            funman_client = Client(SERVER_URL, timeout=None)

            bilayer_path = path.join(
                RESOURCES, "bilayer", "CHIME_SIR_dynamics_BiLayer.json"
            )
            with open(bilayer_path, "r") as bl:
                bilayer_json = json.load(bl)

            infected_threshold = 3

            response = asyncio.run(
                solve_simulation_solve_simulation_put.asyncio_detailed(
                    client=funman_client,
                    json_body=BodySolveSimulationSolveSimulationPut(
                        SimulationScenario.from_dict(
                            {
                                "model": {
                                    "main_fn": "funman_demo.sim.CHIME.CHIME_SIR.main"
                                },
                                "query": {},
                            }
                        )
                    ),
                )
            )
            try:
                result = SimulationScenarioResult.from_dict(
                    src_dict=json.loads(response.content.decode())
                )
                assert result.query_satisfied

            except Exception as e:
                try:
                    result = AnalysisScenarioResultException.from_dict(
                        src_dict=json.loads(response.content.decode())
                    )
                    assert False
                except Exception as e1:
                    assert False


if __name__ == "__main__":
    unittest.main()
