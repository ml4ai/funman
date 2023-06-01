import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from funman.api.api import app, settings
from funman.scenario.consistency import ConsistencyScenario
from funman.scenario.parameter_synthesis import ParameterSynthesisScenario
from funman.server.query import QueryResponse

FILE_DIRECTORY = Path(__file__).resolve().parent
API_BASE_PATH = FILE_DIRECTORY / ".."
RESOURCES = API_BASE_PATH / "resources"
API_SERVER_HOST = "0.0.0.0"
API_SERVER_PORT = 8190
SERVER_URL = f"http://{API_SERVER_HOST}:{API_SERVER_PORT}"
OPENAPI_URL = f"{SERVER_URL}/openapi.json"
CLIENT_NAME = "funman-api-client"

TEST_OUT = FILE_DIRECTORY / "out"
TEST_OUT.mkdir(parents=True, exist_ok=True)

TEST_API_TOKEN = "funman-test-api-token"
settings.funman_api_token = TEST_API_TOKEN


class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = TemporaryDirectory(prefix=f"{cls.__name__}_")

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def setUp(self):
        self.test_dir = Path(self._tmpdir.name) / self._testMethodName
        self.test_dir.mkdir()
        settings.data_path = str(self.test_dir)

    def test_storage(self):
        bilayer_path = (
            RESOURCES / "bilayer" / "CHIME_SIR_dynamics_BiLayer.json"
        )
        with bilayer_path.open() as bl:
            bilayer_json = json.load(bl)
        infected_threshold = 130
        init_values = {"S": 9998, "I": 1, "R": 1}

        first_id = None
        with TestClient(app) as client:
            response = client.post(
                "/queries",
                json={
                    "request": {
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
                },
                headers={"token": f"{TEST_API_TOKEN}"},
            )
            assert response.status_code == 200
            data = QueryResponse.parse_raw(response.content.decode())
            assert data.kind == ConsistencyScenario.get_kind()
            first_id = data.id

        with TestClient(app) as client:
            response = client.get(
                f"/queries/{first_id}", headers={"token": f"{TEST_API_TOKEN}"}
            )
            assert response.status_code == 200
            got_data = QueryResponse.parse_raw(response.content.decode())
            assert first_id == got_data.id
            assert got_data.kind == ConsistencyScenario.get_kind()

    def test_api_bad_token(self):
        with TestClient(app) as client:
            response = client.get(f"/queries/bogus")
            assert response.status_code == 401

    def test_api_bad_id(self):
        with TestClient(app) as client:
            response = client.get(
                f"/queries/bogus", headers={"token": f"{TEST_API_TOKEN}"}
            )
            assert response.status_code == 404

    def test_bilayer_consistency(self):
        bilayer_path = (
            RESOURCES / "bilayer" / "CHIME_SIR_dynamics_BiLayer.json"
        )
        with bilayer_path.open() as bl:
            bilayer_json = json.load(bl)
        infected_threshold = 130
        init_values = {"S": 9998, "I": 1, "R": 1}

        with TestClient(app) as client:
            response = client.post(
                "/queries",
                json={
                    "request": {
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
                },
                headers={"token": f"{TEST_API_TOKEN}"},
            )
            assert response.status_code == 200
            data = QueryResponse.parse_raw(response.content.decode())
            assert data.kind == ConsistencyScenario.get_kind()
            assert data.result

    def test_bilayer_parameter_synthesis(self):
        bilayer_path = (
            RESOURCES / "bilayer" / "CHIME_SIR_dynamics_BiLayer.json"
        )
        with bilayer_path.open() as bl:
            bilayer_json = json.load(bl)
        infected_threshold = 130
        init_values = {"S": 9998, "I": 1, "R": 1}

        lb = 0.000067 * (1 - 0.5)
        ub = 0.000067 * (1 + 0.5)

        with TestClient(app) as client:
            response = client.post(
                "/queries",
                json={
                    "request": {
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
                        "parameters": [
                            {
                                "name": "beta",
                                "lb": lb,
                                "ub": ub,
                                "label": "all",
                            }
                        ],
                    },
                    "config": {"tolerance": 1.0e-8, "number_of_processes": 1},
                },
                headers={"token": f"{TEST_API_TOKEN}"},
            )
            assert response.status_code == 200
            data = QueryResponse.parse_raw(response.content.decode())
            assert data.kind == ParameterSynthesisScenario.get_kind()
            assert data.result
