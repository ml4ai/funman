import os
import unittest

from funman.funman import Funman
from funman.model import Parameter, Model
from funman.scenario import ParameterSynthesisScenario
from funman.scenario import ParameterSynthesisScenarioResult

RESOURCES = os.path.join("resources")
CACHED = os.path.join(RESOURCES, "cached")

class TestCachedParameterSpace(unittest.TestCase):
    def test_cached(self):
        gromet_file1 = "chime1"
        result1 : ParameterSynthesisScenarioResult = Funman().solve(
            ParameterSynthesisScenario(
                [Parameter("beta_0"), Parameter("beta_1")], 
                gromet_file1,
                config = {
                    "epochs": [(0, 20), (20, 60)],
                    "population_size": 1002,
                    "infectious_days": 14.0,
                    # "write_cache_parameter_space" : "ps1_out.json",
                    "read_cache_parameter_space" : os.path.join(CACHED, "parameter_space_1.json"),
                    "real_time_plotting": False
                }))
        assert len(result1.parameter_space.false_boxes) == 16


if __name__ == "__main__":
    unittest.main()
