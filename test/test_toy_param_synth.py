import unittest

from pysmt.shortcuts import GE, LE, And, Real, Symbol
from pysmt.typing import REAL

from funman import Funman
from funman.model import Parameter, QueryTrue
from funman.model2smtlib.translate import Encoder
from funman.model.encoded import EncodedModel
from funman.scenario.parameter_synthesis import ParameterSynthesisScenario
from funman.utils.search_utils import SearchConfig


class TestCompilation(unittest.TestCase):
    def test_toy(self):

        x = Symbol("x", REAL)
        parameters = [Parameter("x", symbol=x)]

        # 0.0 <= x <= 5
        model = EncodedModel(And(LE(x, Real(5.0)), GE(x, Real(0.0))))

        scenario = ParameterSynthesisScenario(
            parameters, model, QueryTrue(), smt_encoder=Encoder()
        )
        funman = Funman()
        config = SearchConfig()
        result = funman.solve(scenario, config)

    def test_toy_2d(self):

        x = Symbol("x", REAL)
        y = Symbol("y", REAL)
        parameters = [Parameter("x", symbol=x), Parameter("y", symbol=y)]

        # 0.0 < x < 5.0, 10.0 < y < 12.0
        model = EncodedModel(
            And(
                LE(x, Real(5.0)),
                GE(x, Real(0.0)),
                LE(y, Real(12.0)),
                GE(y, Real(10.0)),
            )
        )

        scenario = ParameterSynthesisScenario(
            parameters, model, QueryTrue(), smt_encoder=Encoder()
        )
        funman = Funman()
        config = SearchConfig(tolerance=1e-1)
        result = funman.solve(scenario, config=config)
        assert result


if __name__ == "__main__":
    unittest.main()
