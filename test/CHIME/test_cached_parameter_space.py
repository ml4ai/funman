import os
import unittest

from funman import Funman
from funman.model import Parameter, QueryLE
from funman.model.chime import ChimeModel
from funman.scenario.parameter_synthesis import ParameterSynthesisScenario
from funman.scenario.parameter_synthesis import ParameterSynthesisScenarioResult
from funman.search_utils import ResultCombinedHandler, SearchConfig
from funman.examples.chime import CHIME
from funman_demo.handlers import ResultCacheWriter, RealtimeResultPlotter
from model2smtlib.chime.translate import ChimeEncoder

RESOURCES = os.path.join("resources")
CACHED = os.path.join(RESOURCES, "cached")


class TestCachedParameterSpace(unittest.TestCase):
    def test_multiprocessing(self):

        parameters = [
            Parameter("beta_0", lb=0.0, ub=0.5),
            Parameter("beta_1", lb=0.0, ub=0.5),
        ]
        model = ChimeModel(
            init_values={"s": 1000, "i": 1, "r": 1},
            parameter_bounds={
                "beta": [0.00067, 0.00067],
                "gamma": [1.0 / 14.0, 1.0 / 14.0],
            },
            config={
                "epochs": [(0, 20), (20, 30)],
                "population_size": 1002,
                "infectious_days": 14.0,
            },
            chime=CHIME(),
        )
        query = QueryLE("i", 100)
        result1: ParameterSynthesisScenarioResult = Funman().solve(
            ParameterSynthesisScenario(
                parameters, model, query, smt_encoder=ChimeEncoder()
            ),
            SearchConfig(
                tolerance=0.4,
                handler=ResultCombinedHandler(
                    [ResultCacheWriter(os.path.join(CACHED, "example.json"))]
                ),
            ),
        )


if __name__ == "__main__":
    unittest.main()
