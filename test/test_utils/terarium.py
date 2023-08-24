import contextlib
import json
import unittest
from time import sleep

import httpx
from fastapi.testclient import TestClient

from funman.api.api import app


class MockTerariumTestCase(unittest.TestCase):
    """
    TestCase that mimics the functions a user without the funman
    package might need to interact with the API server running
    on Terarium
    """

    def setUp(self) -> None:
        """
        Setup the test case with a running FastAPI TestClient
        """
        with contextlib.ExitStack() as stack:
            self.terarium = stack.enter_context(TestClient(app))
            self._exit_stack = self.terarium.exit_stack.pop_all()

    def tearDown(self) -> None:
        """
        Tear down the TestClient
        """
        self._exit_stack.close()

    def POST(self, url: str, json: dict, *, expect=200):
        """
        Make a POST request through the TestClient
        """
        response = self.terarium.post(url, json=json)
        assert (
            response.status_code == expect
        ), f"Unexpected status code {response.status_code}. Expected {expect}"
        return response

    def GET(self, url: str, *, expect=200):
        """
        Make a GET request through the TestClient
        """
        response = self.terarium.get(url)
        assert (
            response.status_code == expect
        ), f"Unexpected status code {response.status_code}. Expected {expect}"
        return response

    def decode_response_to_dict(self, response: httpx.Response) -> dict:
        """
        Convert the response from the TestClient to a dictionary
        """
        return json.loads(response.content.decode())

    def poll_until_done(self, uuid, sleep_step=1.0, max_steps=20):
        """
        Helper function to poll the status of the request associated
        with the provided UUID.
        """
        while True:
            # Sleep for wait_time
            sleep(sleep_step)
            # Check the status of the query
            response = self.terarium.get(f"/api/queries/{uuid}")
            # Ensure no error status code
            assert (
                response.status_code == 200
            ), f"Response code was not 200: {response.status_code}"
            # Get the results
            results = self.decode_response_to_dict(response)
            assert (
                results.get("error", False) is False
            ), f"Request {uuid} errored during processing"
            # Return if processing is done
            if results.get("done", False):
                return results
            # Track steps
            max_steps -= 1
            assert max_steps > 0

    def post_query(self, model: dict, request: dict) -> str:
        """
        Make a POST request to /api/queries through the TestClient.
        - model: The model to query
        - request: The FunmanWorkRequest

        The response to a query returns a FunmanWorkUnit with fields:
        - id: The UUID assign to the queued FunmanWorkRequest
        - model: A copy of the submitted model
        - request: A copy of the submitted request
        """
        work_unit = self.decode_response_to_dict(
            self.POST("/api/queries", {"model": model, "request": request})
        )
        # Extract the UUID
        return work_unit["id"]

    def get_status(self, uuid: str) -> dict:
        """
        Make a GET request to /api/queries/{uuid} through the TestClient.

        The response to a query returns a FunmanResults with fields:
        - id: The UUID assign to the FunmanWorkRequest
        - model: A copy of the submitted model
        - request: A copy of the submitted request
        - done: A boolean flag for if the request has finished processing
        - error: A boolean flag for if the request errored
        - parameter_space: The current ParameterSpace if one exists
        """
        return self.decode_response_to_dict(self.GET(f"/api/queries/{uuid}"))

    def post_query_and_wait_until_done(
        self, model: dict, request: dict, *, expect_error: bool = False
    ) -> dict:

        """
        Make a query with a provided model and request and wait until it is done processing
        (by polling the status of the request until it reports it is done).
        - model: One of the supported models
        - request: A request to funman
        """
        uuid = self.post_query(model, request)
        self.poll_until_done(uuid)
        results = self.get_status(uuid)

        is_done_processing = results.get("done", False)
        error_occurred = results.get("error", False)

        if expect_error:
            # The results should indicate an error
            assert (
                error_occurred is True
            ), f"An unexpected success occured while processing request with id '{uuid}'"
        else:
            # The results should not indicate an error
            assert (
                error_occurred is False
            ), f"An unexpected error occured while processing request with id '{uuid}'"
            # Processing should be done
            assert (
                is_done_processing is True
            ), f"Expected work to be done for {uuid}"
        return results
