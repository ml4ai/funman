import os
import unittest

from funman.funman import Funman
from funman.model import Parameter
from funman.scenario import ParameterSynthesisScenario
from funman.scenario import ParameterSynthesisScenarioResult
from funman.search_utils import ResultCombinedHandler, SearchConfig

from funman_demo.handlers import ResultCacheWriter, RealtimeResultPlotter

RESOURCES = os.path.join("resources")
CACHED = os.path.join(RESOURCES, "cached")

class TestCachedParameterSpace(unittest.TestCase):
    
    def test_multiprocessing(self):
        gromet_file1 = "chime1"

        parameters = [Parameter("beta_0", lb=0.0, ub=0.5), Parameter("beta_1", lb=0.0, ub=0.5)]
        result1 : ParameterSynthesisScenarioResult = Funman().solve(
            ParameterSynthesisScenario(
                parameters,
                gromet_file1,
                config = {
                    "epochs": [(0, 20), (20, 30)],
                    "population_size": 1002,
                    "infectious_days": 14.0
                }),
            SearchConfig(
                tolerance=0.4,
                handler = ResultCombinedHandler([
                    ResultCacheWriter(os.path.join(CACHED, "search_cache.json"))
                ]),
            ))


if __name__ == "__main__":
    unittest.main()
