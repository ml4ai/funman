import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from funman.api.api import app, settings
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
        settings.data_path = self.test_dir

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
            first_id = data.id

        with TestClient(app) as client:
            response = client.get(
                f"/queries/{first_id}", headers={"token": f"{TEST_API_TOKEN}"}
            )
            assert response.status_code == 200
            get_data = QueryResponse.parse_raw(response.content.decode())
            assert first_id == get_data.id

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
