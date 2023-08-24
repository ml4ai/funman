import json
import unittest
from pathlib import Path

from test_utils.terarium import MockTerariumTestCase

# Read in the model associated with this example
PETRINET_MODEL = json.loads(
    Path("resources/amr/petrinet/mira/models/scenario1_a.json").read_bytes()
)


class TestPetriNet_Example1(MockTerariumTestCase):
    def test_default(self):
        """
        Run with the default request behavior
        """
        results = self.post_query_and_wait_until_done(PETRINET_MODEL, {})
        parameter_space = results["parameter_space"]
        # TODO check results

    @unittest.skip(reason="TODO")
    def test_with_parameters(self):
        """
        Run with a few parameters specified
        """
        results = self.query(
            PETRINET_MODEL, {"parameters": [{"name": "beta"}]}
        )
        print(json.dumps(results.parameter_space.dict(), indent=2))

    @unittest.skip(reason="TODO")
    def test_with_parameter_bounds(self):
        """
        Run with a few parameter bounds specified
        """
        results = self.query(PETRINET_MODEL, {})
        print(json.dumps(results.parameter_space.dict(), indent=2))

    @unittest.skip(reason="TODO")
    def test_with_parameters_and_bounds(self):
        """
        Run with a few parameters specified
        """
        results = self.query(PETRINET_MODEL, {})
        print(json.dumps(results.parameter_space.dict(), indent=2))
