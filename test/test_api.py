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

AMR_DIR = RESOURCES / "amr"
AMR_EXAMPLES_PETRI_DIR = AMR_DIR / "petrinet" / "amr-examples"
AMR_EXAMPLES_REGNET_DIR = AMR_DIR / "regnet" / "amr-examples"

SKEMA_PETRI_DIR = AMR_DIR / "petrinet" / "skema"
SKEMA_REGNET_DIR = AMR_DIR / "regnet" / "skema"

MIRA_PETRI_DIR = AMR_DIR / "petrinet" / "mira"
MIRA_PETRI_MODELS = MIRA_PETRI_DIR / "models"
MIRA_PETRI_REQUESTS = MIRA_PETRI_DIR / "requests"

TEST_OUT = FILE_DIRECTORY / "out"
TEST_OUT.mkdir(parents=True, exist_ok=True)

TEST_API_TOKEN = "funman-test-api-token"
TEST_BASE_URL = "funman"


class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        settings.funman_api_token = TEST_API_TOKEN
        settings.funman_base_url = TEST_BASE_URL
        cls._tmpdir = TemporaryDirectory(prefix=f"{cls.__name__}_")

    @classmethod
    def tearDownClass(cls) -> None:
        settings.funman_api_token = None
        settings.funman_base_url = None
        cls._tmpdir.cleanup()

    def setUp(self):
        self.test_dir = Path(self._tmpdir.name) / self._testMethodName
        self.test_dir.mkdir()
        settings.data_path = str(self.test_dir)

    def tearDown(self):
        settings.data_path = "."

    def wait_for_done(self, client, id, wait_time=1.0, steps=20):
        while True:
            sleep(wait_time)
            response = client.get(
                f"/api/queries/{id}", headers={"token": f"{TEST_API_TOKEN}"}
            )
            assert (
                response.status_code == 200
            ), f"Response code was not 200: {response.status_code}"
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
            assert (
                response.status_code == 200
            ), f"Response code was not 200: {response.status_code}"
            work_unit = FunmanWorkUnit.parse_raw(response.content.decode())
            first_id = work_unit.id
            data = self.wait_for_done(client, first_id)
            self.check_consistency_success(data.parameter_space)

        with TestClient(app) as client:
            response = client.get(
                f"/api/queries/{first_id}",
                headers={"token": f"{TEST_API_TOKEN}"},
            )
            assert (
                response.status_code == 200
            ), f"Response code was not 200: {response.status_code}"
            got_data = FunmanResults.parse_raw(response.content.decode())
            assert first_id == got_data.id
            self.check_consistency_success(data.parameter_space)

    def test_api_bad_token(self):
        with TestClient(app) as client:
            response = client.get("/api/queries/bogus")
            assert (
                response.status_code == 401
            ), f"Response code was not 401: {response.status_code}"

    def test_api_bad_id(self):
        with TestClient(app) as client:
            response = client.get(
                "/api/queries/bogus", headers={"token": f"{TEST_API_TOKEN}"}
            )
            assert (
                response.status_code == 404
            ), f"Response code was not 404: {response.status_code}"

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
                        "structure_parameters": [
                            {
                                "name": "num_steps",
                                "lb": 3.0,
                                "ub": 3.0,
                                "label": "any",
                            },
                            {
                                "name": "step_size",
                                "lb": 1.0,
                                "ub": 1.0,
                                "label": "any",
                            },
                        ],
                    },
                },
                headers={"token": f"{TEST_API_TOKEN}"},
            )
            assert (
                response.status_code == 200
            ), f"Response code was not 200: {response.status_code}"
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
                            {
                                "name": "num_steps",
                                "lb": 3.0,
                                "ub": 3.0,
                                "label": "any",
                            },
                            {
                                "name": "step_size",
                                "lb": 1.0,
                                "ub": 1.0,
                                "label": "any",
                            },
                        ],
                        "config": {
                            "tolerance": 1.0e-8,
                            "number_of_processes": 1,
                        },
                    },
                },
                headers={"token": f"{TEST_API_TOKEN}"},
            )
            assert (
                response.status_code == 200
            ), f"Response code was not 200: {response.status_code}"
            work_unit = FunmanWorkUnit.parse_raw(response.content.decode())
            data = self.wait_for_done(client, work_unit.id)
            self.check_parameter_synthesis_success(data.parameter_space)

    def test_amr_pairs(self):
        pairs = [
            # (model, request)
            (
                AMR_EXAMPLES_PETRI_DIR / "sir.json",
                AMR_EXAMPLES_PETRI_DIR / "sir_request1.json",
            ),
            # (
            #     MIRA_PETRI_MODELS / "scenario2_a_beta_scale_static.json",
            #     MIRA_PETRI_REQUESTS
            #     / "request2_b_default_w_compartmental_constrs.json",
            # ),
            # (
            #     MIRA_PETRI_MODELS / "scenario2_a_beta_scale_static.json",
            #     MIRA_PETRI_REQUESTS
            #     / "request2_b_default_wo_compartmental_constrs.json",
            # ),
            # (
            #     MIRA_PETRI_MODELS / "scenario2_a_beta_scale_static_fixed.json",
            #     MIRA_PETRI_REQUESTS
            #     / "request2_b_default_w_compartmental_constrs.json",
            # ),
        ]
        for model_path, request_path in pairs:
            msg = f"({model_path.name}, {request_path.name})"
            with self.subTest(msg):
                self.subtest_amr_pair(model_path, request_path)

    def subtest_amr_pair(self, model_path, request_path):
        model = json.loads(model_path.read_bytes())
        request = json.loads(request_path.read_bytes())
        with TestClient(app) as client:
            response = client.post(
                "/api/queries",
                json={"model": model, "request": request},
                headers={"token": f"{TEST_API_TOKEN}"},
            )
            assert (
                response.status_code == 200
            ), f"Response code was not 200: {response.status_code}"
            work_unit = FunmanWorkUnit.parse_raw(response.content.decode())
            data = self.wait_for_done(client, work_unit.id)
            assert data, "No FunmanResults returned"
            assert (
                data.error is False
            ), "FunmanResults error flag is set. Worker processing error."
            assert (
                data.parameter_space is not None
            ), "FunmanResults has no ParameterSpace"
