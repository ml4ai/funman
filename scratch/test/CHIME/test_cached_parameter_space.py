import logging
import os
import unittest

from funman_demo.example.chime import CHIME
from funman_demo.handlers import ResultCacheWriter

from funman import Funman
from funman.model import ModelParameter
from funman.model.encoded import EncodedModel
from funman.representation.representation import (
    ResultCombinedHandler,
    SearchConfig,
)
from funman.scenario.parameter_synthesis import ParameterSynthesisScenario
from funman.search import BoxSearch

l = logging.getLogger(__file__)
l.setLevel(logging.ERROR)

RESOURCES = os.path.join("resources")
CACHED = os.path.join(RESOURCES, "cached")


class TestCachedParameterSpace(unittest.TestCase):
    def test_cached_parameter_space(self):
        chime = CHIME()
        vars, (parameters, init, dynamics, query) = chime.make_model(
            assign_betas=False
        )
        num_timepoints = 5
        phi = chime.encode_time_horizon(
            parameters, init, dynamics, [], num_timepoints
        )
        (
            susceptible,
            infected,
            recovered,
            susceptible_n,
            infected_n,
            recovered_n,
            scale,
            betas,
            gamma,
            n,
            delta,
        ) = vars
        parameters = [ModelParameter(name="beta", _symbol=betas[0])]

        model = EncodedModel(phi)

        scenario = ParameterSynthesisScenario(
            parameters,
            model,
            chime.encode_query_time_horizon(query, num_timepoints),
            _search=BoxSearch(),
        )
        funman = Funman()
        config = SearchConfig(
            tolerance=1e-1,
            queue_timeout=10,
            handler=ResultCombinedHandler(
                [ResultCacheWriter(os.path.join(CACHED, "example.json"))]
            ),
        )
        result = funman.solve(scenario, config=config)
        l.info(f"True Boxes: {result.parameter_space.true_boxes}")
        assert result.parameter_space


if __name__ == "__main__":
    unittest.main()
