import unittest

from pysmt.shortcuts import GE, LE, And, Real, Symbol
from pysmt.typing import REAL

from funman import Funman, FUNMANConfig
from funman.model import EncodedModel, QueryTrue
from funman.representation.representation import ModelParameter
from funman.scenario.parameter_synthesis import ParameterSynthesisScenario
from funman.translate import EncodedEncoder


class TestCompilation(unittest.TestCase):
    def test_toy(self):
        parameters = [ModelParameter(name="x")]
        x = parameters[0].symbol()

        # 0.0 <= x <= 5
        model = EncodedModel()
        model.parameters = parameters
        model._formula = And(LE(x, Real(5.0)), GE(x, Real(0.0)))

        scenario = ParameterSynthesisScenario(
            parameters=parameters,
            model=model,
            query=QueryTrue(),
        )
        funman = Funman()
        config = FUNMANConfig(
            number_of_processes=1,
            substitute_subformulas=False,
            normalization_constant=5.0,
        )
        result = funman.solve(scenario, config)

    def test_toy_2d(self):
        parameters = [
            ModelParameter(name="x"),
            ModelParameter(name="y"),
        ]
        x = parameters[0].symbol()
        y = parameters[1].symbol()

        # 0.0 < x < 5.0, 10.0 < y < 12.0
        model = EncodedModel()
        model.parameters = parameters
        model._formula = And(
            LE(x, Real(5.0)),
            GE(x, Real(0.0)),
            LE(y, Real(12.0)),
            GE(y, Real(10.0)),
        )

        scenario = ParameterSynthesisScenario(
            parameters=parameters,
            model=model,
            query=QueryTrue(),
        )
        funman = Funman()
        config = FUNMANConfig(
            tolerance=1e-1,
            number_of_processes=1,
            substitute_subformulas=False,
            normalization_constant=12.0,
        )
        result = funman.solve(scenario, config=config)
        assert result


if __name__ == "__main__":
    unittest.main()
