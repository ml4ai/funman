import logging
import unittest

from pysmt.shortcuts import (
    FALSE,
    GE,
    GT,
    LE,
    LT,
    TRUE,
    And,
    Equals,
    ForAll,
    Function,
    FunctionType,
    Iff,
    Int,
    Plus,
    Real,
    Symbol,
    get_model,
    simplify,
    substitute,
)
from pysmt.typing import BOOL, INT, REAL

from funman import Funman
from funman.examples.chime import CHIME
from funman.model import Parameter
from funman.model.encoded import EncodedModel
from funman.scenario.parameter_synthesis import ParameterSynthesisScenario
from funman.search import BoxSearch
from funman.utils.search_utils import SearchConfig

l = logging.getLogger(__file__)
l.setLevel(logging.ERROR)


class TestCompilation(unittest.TestCase):
    def test_chime(self):
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
        parameters = [Parameter("beta", symbol=betas[0])]

        model = EncodedModel(phi)

        scenario = ParameterSynthesisScenario(
            parameters,
            model,
            chime.encode_query_time_horizon(query, num_timepoints),
            search=BoxSearch(),
        )
        funman = Funman()
        config = SearchConfig(tolerance=1e-1, queue_timeout=10)
        result = funman.solve(scenario, config=config)
        l.info(f"True Boxes: {result.parameter_space.true_boxes}")
        assert result.parameter_space


if __name__ == "__main__":
    unittest.main()
