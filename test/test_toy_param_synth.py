import sys
sys.path.append('/Users/dmosaphir/SIFT/Projects/ASKEM/code/funman/src')
from funman.search import BoxSearch, SearchConfig
from funman.constants import NEG_INFINITY, POS_INFINITY
from pysmt.shortcuts import (
    get_model,
    And,
    Symbol,
    FunctionType,
    Function,
    Equals,
    Int,
    Real,
    substitute,
    TRUE,
    FALSE,
    Iff,
    Plus,
    ForAll,
    LT,
    simplify,
    GT,
    LE,
    GE,
)
from pysmt.typing import INT, REAL, BOOL
import unittest
import os
from funman import Funman
from funman.model import Parameter, EncodedModel
from funman.scenario import ParameterSynthesisScenario
from funman.math_utils import lt
from funman.search_utils import Interval,Box
from funman.parameter_space import ParameterSpace

class TestCompilation(unittest.TestCase):
    def test_toy(self):

        x = Symbol("x", REAL)
        parameters = [Parameter("x", x)]

        # 0.0 <= x <= 5
        model = EncodedModel(And(LE(x, Real(5.0)), GE(x, Real(0.0))))

        scenario = ParameterSynthesisScenario(parameters, model, BoxSearch())
        funman = Funman()
        config = SearchConfig()
        result = funman.solve(scenario, config)

    def test_toy_2d(self):

        x = Symbol("x", REAL)
        y = Symbol("y", REAL)
        parameters = [Parameter("x", x), Parameter("y", y)]

        # 0.0 < x < 5.0, 10.0 < y < 12.0
        model = EncodedModel(
            And(
                LE(x, Real(5.0)), GE(x, Real(0.0)), LE(y, Real(12.0)), GE(y, Real(10.0))
            )
        )

        scenario = ParameterSynthesisScenario(parameters, model, BoxSearch())
        funman = Funman()
        config = SearchConfig(tolerance=1e-1)
        result = funman.solve(scenario, config=config)
        assert result


if __name__ == "__main__":
    unittest.main()
