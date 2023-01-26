import logging
import unittest

from funman_demo.example.chime import CHIME
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
from funman.model import Parameter, QueryEncoded
from funman.model.encoded import EncodedModel
from funman.scenario.parameter_synthesis import ParameterSynthesisScenario
from funman.search import BoxSearch, SearchConfig

l = logging.getLogger(__file__)
l.setLevel(logging.INFO)


class TestCompilation(unittest.TestCase):
    def test_chime(self):
        chime = CHIME()
        vars, (parameters, init, dynamics, query) = chime.make_model(
            assign_betas=False
        )
        num_timepoints = 2
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
        parameters = [
            Parameter(name="beta", lb=1e-6, ub=1e-4, _symbol=betas[0])
        ]

        model = EncodedModel(phi)
        query = QueryEncoded()
        query._formula = chime.encode_query_time_horizon(query, num_timepoints)
        scenario = ParameterSynthesisScenario(
            parameters,
            model,
            query,
            _search=BoxSearch(),
        )
        funman = Funman()
        config = SearchConfig(tolerance=1e-7, queue_timeout=10)
        result = funman.solve(scenario, config=config)
        l.info(f"True Boxes: {result.parameter_space.true_boxes}")
        assert result.parameter_space


if __name__ == "__main__":
    unittest.main()
