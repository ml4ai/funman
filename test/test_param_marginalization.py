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
from funman.model import EncodedModel, Parameter, QueryTrue
from funman.scenario import ParameterSynthesisScenario
from funman.search import SearchConfig
from funman.translate import EncodedEncoder


class TestCompilation(unittest.TestCase):
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
            parameters, model, QueryTrue(), smt_encoder=EncodedEncoder()
        )
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
                ans = boxes_of_interest[i1].check_bounds_disjoint_equal(
                    boxes_of_interest[i2]
                )
                if ans[0] == False:
                    print(ans)


if __name__ == "__main__":
    unittest.main()
