import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from time import sleep

from fastapi.testclient import TestClient

from funman.api.api import app, settings
from funman.representation.representation import ParameterSpace
from funman.server.query import FunmanResults, FunmanWorkUnit

FILE_DIRECTORY = Path(__file__).resolve().parent
API_BASE_PATH = FILE_DIRECTORY / ".."
RESOURCES = API_BASE_PATH / "resources"

TEST_OUT = FILE_DIRECTORY / "out"
TEST_OUT.mkdir(parents=True, exist_ok=True)

TEST_API_TOKEN = "funman-test-api-token"
TEST_BASE_URL = "funman"
settings.funman_api_token = TEST_API_TOKEN
settings.funman_base_url = TEST_BASE_URL


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

    def wait_for_done(self, client, id, wait_time=1.0, steps=20):
        while True:
            sleep(wait_time)
            response = client.get(
                f"/api/queries/{id}", headers={"token": f"{TEST_API_TOKEN}"}
            )
            assert response.status_code == 200
            data = FunmanResults.parse_raw(response.content.decode())
            if data.done:
                return data
            steps -= 1
            assert steps > 0

    def check_consistency_success(self, parameter_space: ParameterSpace):
        assert parameter_space is not None
        assert len(parameter_space.true_boxes) == 0
        assert len(parameter_space.false_boxes) == 0
        assert len(parameter_space.true_points) == 1
        assert len(parameter_space.false_points) == 0

    def check_parameter_synthesis_success(
        self, parameter_space: ParameterSpace
    ):
        assert parameter_space is not None
        assert len(parameter_space.true_boxes) > 0

    def test_storage(self):
        bilayer_path = (
            RESOURCES / "bilayer" / "CHIME_SIR_dynamics_BiLayer.json"
        )
        with bilayer_path.open() as bl:
            bilayer_json = json.load(bl)
        infected_threshold = 130
        init_values = {"S": 9998, "I": 1, "R": 1}

        lb = 0.000067 * (1 - 0.5)
        ub = 0.000067 * (1 + 0.5)

        first_id = None
        with TestClient(app) as client:
            response = client.post(
                "/api/queries",
                json={
                    "model": {
                        "init_values": init_values,
                        "bilayer": {"json_graph": bilayer_json},
                        "parameter_bounds": {
                            "beta": [lb, ub],
                            "gamma": [1.0 / 14.0, 1.0 / 14.0],
                        },
                    },
                    "request": {
                        "query": {
                            "variable": "I",
                            "ub": infected_threshold,
                            "at_end": False,
                        },
                    },
                },
                headers={"token": f"{TEST_API_TOKEN}"},
            )
            assert response.status_code == 200
            work_unit = FunmanWorkUnit.parse_raw(response.content.decode())
            first_id = work_unit.id
            data = self.wait_for_done(client, first_id)
            self.check_consistency_success(data.parameter_space)

        with TestClient(app) as client:
            response = client.get(
                f"/api/queries/{first_id}",
                headers={"token": f"{TEST_API_TOKEN}"},
            )
            assert response.status_code == 200
            got_data = FunmanResults.parse_raw(response.content.decode())
            assert first_id == got_data.id
            self.check_consistency_success(data.parameter_space)

    def test_api_bad_token(self):
        with TestClient(app) as client:
            response = client.get(f"/api/queries/bogus")
            assert response.status_code == 401

    def test_api_bad_id(self):
        with TestClient(app) as client:
            response = client.get(
                f"/api/queries/bogus", headers={"token": f"{TEST_API_TOKEN}"}
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
        scale_factor = 0.5
        lb = 0.000067 * (1 - scale_factor)
        ub = 0.000067 * (1 + scale_factor)

        with TestClient(app) as client:
            response = client.post(
                "/api/queries",
                json={
                    "model": {
                        "init_values": init_values,
                        "bilayer": {"json_graph": bilayer_json},
                                    "parameter_bounds" :{
                "beta": [lb, ub],
                "gamma": [1.0 / 14.0, 1.0 / 14.0],
            },
                    },
                    "request": {
                        "query": {
                            "variable": "I",
                            "ub": infected_threshold,
                            "at_end": False,
                        },
                        
                        "structure_parameters": [
                            {'name': 'num_steps', 'lb': 3.0, 'ub': 3.0, 'label': 'any'}, 
                            {'name': 'step_size', 'lb': 1.0, 'ub': 1.0, 'label': 'any'}
                        ]
                    },
                },
                headers={"token": f"{TEST_API_TOKEN}"},
            )
            assert response.status_code == 200
            work_unit = FunmanWorkUnit.parse_raw(response.content.decode())
            data = self.wait_for_done(client, work_unit.id)
            self.check_consistency_success(data.parameter_space)

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
                "/api/queries",
                json={
                    "model": {
                        "init_values": init_values,
                        "bilayer": {"json_graph": bilayer_json},
                        "parameter_bounds": {
                            "beta": [lb, ub],
                            "gamma": [1.0 / 14.0, 1.0 / 14.0],
                        },
                    },
                    "request": {
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
                        "structure_parameters": [
                            {'name': 'num_steps', 'lb': 3.0, 'ub': 3.0, 'label': 'any'}, 
                            {'name': 'step_size', 'lb': 1.0, 'ub': 1.0, 'label': 'any'}
                        ],
                        "config": {
                            "tolerance": 1.0e-8,
                            "number_of_processes": 1,
                        },
                    },
                },
                headers={"token": f"{TEST_API_TOKEN}"},
            )
            assert response.status_code == 200
            work_unit = FunmanWorkUnit.parse_raw(response.content.decode())
            data = self.wait_for_done(client, work_unit.id)
            self.check_parameter_synthesis_success(data.parameter_space)

    def test_amr_petri_net(self):
        # Alternative example
        EXAMPLE_DIR = RESOURCES / "amr" / "petrinet" / "amr-examples"
        MODEL_PATH = EXAMPLE_DIR / "sir.json"
        REQUEST_PATH = EXAMPLE_DIR / "sir_request1.json"
        model = json.loads(MODEL_PATH.read_bytes())
        request = json.loads(REQUEST_PATH.read_bytes())
        with TestClient(app) as client:
            response = client.post(
                "/api/queries",
                json={"model": model, "request": request},
                headers={"token": f"{TEST_API_TOKEN}"},
            )
            assert response.status_code == 200
            work_unit = FunmanWorkUnit.parse_raw(response.content.decode())
            data = self.wait_for_done(client, work_unit.id)
            assert data
