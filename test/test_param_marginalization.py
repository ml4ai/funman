import sys

sys.path.append("/Users/dmosaphir/SIFT/Projects/ASKEM/code/funman/src")
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
from funman.scenario.parameter_synthesis import ParameterSynthesisScenario
from funman.search_utils import Box


class TestCompilation(unittest.TestCase):
    def test_toy_2d(self):

        x = Symbol("x", REAL)
        y = Symbol("y", REAL)
        parameters = [Parameter("x", x), Parameter("y", y)]

        # 0.0 < x < 5.0, 10.0 < y < 12.0
        model = EncodedModel(
            And(
                LE(x, Real(5.0)),
                GE(x, Real(0.0)),
                LE(y, Real(12.0)),
                GE(y, Real(10.0)),
            )
        )

        scenario = ParameterSynthesisScenario(parameters, model, BoxSearch())
        funman = Funman()
        config = SearchConfig(tolerance=1e-1)
        result = funman.solve(scenario, config=config)
        #        for b1 in result.parameter_space.true_boxes:
        #            print(Box.union)
        boxes_of_interest = result.parameter_space.true_boxes
        ### Choose sample boxes
        #        box_0 = boxes_of_interest[0]
        #        box_1 = boxes_of_interest[1]
        #        print(Box.check_bounds_disjoint_equal(box_0, box_1))
        for i1 in range(len(boxes_of_interest)):
            for i2 in range(i1 + 1, len(boxes_of_interest)):
                ans = Box.check_bounds_disjoint_equal(
                    boxes_of_interest[i1], boxes_of_interest[i2]
                )
                if ans[0] == False:
                    print(ans)


if __name__ == "__main__":
    unittest.main()
