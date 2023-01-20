import unittest

from pysmt.shortcuts import GE, LE, And, Real, Symbol
from pysmt.typing import REAL

from funman import Funman
from funman.funman import FUNMANConfig
from funman.model import EncodedModel, QueryTrue
from funman.representation.representation import Parameter
from funman.scenario.parameter_synthesis import ParameterSynthesisScenario
from funman.translate import EncodedEncoder


class TestCompilation(unittest.TestCase):
    def test_toy(self):

        x = Symbol("x", REAL)
        parameters = [Parameter(name="x", _symbol=x)]

        # 0.0 <= x <= 5
        model = EncodedModel(_formula=And(LE(x, Real(5.0)), GE(x, Real(0.0))))

        scenario = ParameterSynthesisScenario(
            parameters=parameters,
            model=model,
            query=QueryTrue(),
            _smt_encoder=EncodedEncoder(),
        )
        funman = Funman()
        config = FUNMANConfig()
        result = funman.solve(scenario, config)

    def test_toy_2d(self):

        x = Symbol("x", REAL)
        y = Symbol("y", REAL)
        parameters = [
            Parameter(name="x", _symbol=x),
            Parameter(name="y", _symbol=y),
        ]

        # 0.0 < x < 5.0, 10.0 < y < 12.0
        model = EncodedModel(
            _formula=And(
                LE(x, Real(5.0)),
                GE(x, Real(0.0)),
                LE(y, Real(12.0)),
                GE(y, Real(10.0)),
            )
        )

        scenario = ParameterSynthesisScenario(
            parameters=parameters,
            model=model,
            query=QueryTrue(),
            _smt_encoder=EncodedEncoder(),
        )
        funman = Funman()
        config = FUNMANConfig(tolerance=1e-1)
        result = funman.solve(scenario, config=config)
        assert result


if __name__ == "__main__":
    unittest.main()
