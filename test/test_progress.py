import json
import unittest
from pathlib import Path
from time import sleep

from fastapi.testclient import TestClient

from funman.api.api import app, settings
from funman.funman import Funman, FUNMANConfig
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


class TestProgress(unittest.TestCase):
    def test_progress_via_api(self):
        """
        Run subtest_progress for each of the pairs
        """
        pairs = [
            # (model, request)
            (
                MIRA_PETRI_MODELS / "scenario2_a_beta_scale_static.json",
                MIRA_PETRI_REQUESTS / "request2_b_synthesize.json",
            ),
        ]
        for model_path, request_path in pairs:
            msg = f"({model_path.name}, {request_path.name})"
            with self.subTest(msg):
                self.subtest_progress_via_api(model_path, request_path)

    def subtest_progress_via_api(self, model_path, request_path):
        """
        Check that progress:
        - starts at 0.0
        - only increases or stays the same
        - does not exceed 1.0
        - ends ar 1.0 when done is True
        """
        # Load the input files
        model = json.loads(model_path.read_bytes())
        request = json.loads(request_path.read_bytes())

        # Start a test API client
        with TestClient(app) as client:
            # Make the initial query
            response = client.post(
                "/api/queries",
                json={"model": model, "request": request},
            )
            # Ensure the response code reports success
            assert (
                response.status_code == 200
            ), f"Response code was not 200: {response.status_code}"

            # Parse and extract the work id
            work_unit = FunmanWorkUnit.parse_raw(response.content.decode())
            work_id = work_unit.id

            # Check that the progress starts at 0.0
            progress = work_unit.progress
            assert round(progress, 5) == 0.0, "Progress did not start at 0.0"
            prev_progress = progress

            # Check the status of the query several times while sleeping between
            steps = 20
            while True:
                # Wait
                sleep(1.0)
                # Get status
                response = client.get(f"/api/queries/{work_id}")
                # Ensure success response
                assert (
                    response.status_code == 200
                ), f"Response code was not 200: {response.status_code}"
                # Parse data to a FunmanResults object
                data = FunmanResults.parse_raw(response.content.decode())
                prev_progress = progress
                progress = data.progress

                assert (
                    progress >= 0.0
                ), f"Progress is less than 0.0: {progress}"
                assert (
                    progress <= 1.0
                ), f"Progress is greater than 1.0: {progress}"
                assert (
                    progress >= prev_progress
                ), f"Progress decreased from {prev_progress} to {progress}"
                if data.done:
                    break
                # Limit steps to done
                steps -= 1
                assert steps > 0, "Failed to finish in allowed time"

            # Ensure progress is 1.0 after processing is done
            assert round(progress, 5) == 1.0, "Progress did not end at 1.0"


if __name__ == "__main__":
    unittest.main()
