import contextlib
import json
import sys
import threading
import time
import unittest
from os import chdir, path

import openapi_python_client
import uvicorn
from openapi_python_client import Config as OPCConfig
from openapi_python_client import MetaType

from funman.api.api import app
from funman.model.bilayer import BilayerGraph

RESOURCES = path.join(path.dirname(path.abspath(__file__)), "../resources")
API_BASE_PATH = path.join(path.dirname(path.abspath(__file__)), "..")


class Server(uvicorn.Server):
    def install_signal_handlers(self):
        pass

    @contextlib.contextmanager
    def run_in_thread(self):
        thread = threading.Thread(target=self.run)
        thread.start()
        try:
            while not self.started:
                time.sleep(1e-3)
            yield
        finally:
            self.should_exit = True
            thread.join()


class TestAPI(unittest.TestCase):
    def test_api(self):
        # Start API Server
        config = uvicorn.Config(
            app, host="0.0.0.0", port=8190, log_level="info"
        )
        server = Server(config=config)
        with server.run_in_thread():
            # Server is started.

            # Regenerate api client
            client_path = path.join(API_BASE_PATH, "funman-api-client")
            chdir(API_BASE_PATH)
            if path.exists(client_path):
                openapi_python_client.update_existing_client(
                    url="http://0.0.0.0:8190/openapi.json",
                    path=None,
                    meta=MetaType.POETRY,
                    config=OPCConfig(),
                )
            else:
                openapi_python_client.create_new_client(
                    url="http://0.0.0.0:8190/openapi.json",
                    path=None,
                    meta=MetaType.POETRY,
                    config=OPCConfig(),
                )
            sys.path.append(path.join(API_BASE_PATH, "funman-api-client"))

            from funman_api_client import Client
            from funman_api_client.api.default import solve_solve_put
            from funman_api_client.models import (
                BilayerDynamics,
                BilayerDynamicsJsonGraph,
                BilayerModel,
                BilayerModelInitValues,
                BodySolveSolvePut,
                Config,
                ConsistencyScenario,
                ConsistencyScenarioResult,
                QueryLE,
            )
            from funman_api_client.types import Response

            funman_client = Client("http://localhost:8190")

            bilayer_path = path.join(
                RESOURCES, "bilayer", "CHIME_SIR_dynamics_BiLayer.json"
            )
            with open(bilayer_path, "r") as bl:
                bilayer_json = json.load(bl)
            infected_threshold = 130
            init_values = {"S": 9998, "I": 1, "R": 1}

            my_data: ConsistencyScenarioResult = solve_solve_put.sync_detailed(
                client=funman_client,
                json_body=BodySolveSolvePut(
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
            assert my_data


if __name__ == "__main__":
    unittest.main()
