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
        parameters = [Parameter(name="x")]
        x = parameters[0].symbol()

        # 0.0 <= x <= 5
        model = EncodedModel(_formula=And(LE(x, Real(5.0)), GE(x, Real(0.0))))

        scenario = ParameterSynthesisScenario(
            parameters=parameters,
            model=model,
            query=QueryTrue(),
        )
        funman = Funman()
        config = FUNMANConfig(number_of_processes=1)
        result = funman.solve(scenario, config)

    def test_toy_2d(self):
        parameters = [
            Parameter(name="x"),
            Parameter(name="y"),
        ]
        x = parameters[0].symbol()
        y = parameters[1].symbol()

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
        )
        funman = Funman()
        config = FUNMANConfig(tolerance=1e-1, number_of_processes=1)
        result = funman.solve(scenario, config=config)
        assert result


if __name__ == "__main__":
    unittest.main()
