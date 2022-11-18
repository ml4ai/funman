import sys
sys.path.append('/Users/dmosaphir/SIFT/Projects/ASKEM/code/funman/src')
import matplotlib.pyplot as plt
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
from funman.funman import Funman
from funman.model import Parameter, Model
from funman.scenario import ParameterSynthesisScenario
from funman.math_utils import lt
from funman.search_utils import Interval,Box
from funman.parameter_space import ParameterSpace
from funman.plotting import BoxPlotter

class TestCompilation(unittest.TestCase):
#    def test_toy(self):
#
#        x = Symbol("x", REAL)
#        parameters = [Parameter("x", x)]
#
#        # 0.0 <= x <= 5
#        model = Model(And(LE(x, Real(5.0)), GE(x, Real(0.0))))
#
#        scenario = ParameterSynthesisScenario(parameters, model, BoxSearch())
#        funman = Funman()
#        config = SearchConfig()
#        result = funman.solve(scenario, config)
#
#    def test_toy_2d(self):
#
#        x = Symbol("x", REAL)
#        y = Symbol("y", REAL)
#        parameters = [Parameter("x", x), Parameter("y", y)]
#
#        # 0.0 < x < 5.0, 10.0 < y < 12.0
#        model = Model(
#            And(
#                LE(x, Real(5.0)), GE(x, Real(0.0)), LE(y, Real(12.0)), GE(y, Real(10.0))
#            )
#        )
#
#        scenario = ParameterSynthesisScenario(parameters, model, BoxSearch())
#        funman = Funman()
#        config = SearchConfig(tolerance=1e-1)
#        result = funman.solve(scenario, config=config)
######## begin interval demos
#        a1 = Interval.make_interval([0,2])
#        a2 = Interval.make_interval([1,3])
#        print(Interval.midpoint(a1))
#        print(Interval.union(a1,a2))
####### begin box demos
#        for b1 in result.parameter_space.true_boxes:
#            for b2 in result.parameter_space.false_boxes:
#                print(Box.intersect_two_boxes(b1,b2))
#        print(ParameterSpace._intersect_boxes(result.parameter_space.true_boxes, result.parameter_space.false_boxes))
#        print(ParameterSpace._symmetric_difference(result.parameter_space.true_boxes, result.parameter_space.false_boxes))
#        for b1 in result.parameter_space.true_boxes:
#            for b2 in result.parameter_space.false_boxes:
#                print(Box.union(b1,b2))
#        print(ParameterSpace._union_boxes(boxes_of_interest))
#        boxes_of_interest = result.parameter_space.true_boxes
#        for b1 in boxes_of_interest:
#            print(Box.union)
#        for i1 in range(len(boxes_of_interest)):
#            for i2 in range(i1+1, len(boxes_of_interest)):
#                ans = Box.check_bounds_disjoint_equal(boxes_of_interest[i1], boxes_of_interest[i2])
#                if ans[0] == False:
#                    print(ans)
#                    ## Plot original boxes
#                    BoxPlotter.plot2DBoxes_temp([boxes_of_interest[i1], boxes_of_interest[i2]])
#                    ## Plot heights
#                    bounds = ans[1]
#                    x_bounds = bounds[0]
#                    heights = bounds[1]
#                    y_bounds = bounds[2]
##                    BoxPlotter.plot2DBoxesByHeight_temp(x_bounds, heights)
##                    BoxPlotter.plot2DBoxesByBounds_temp(x_bounds, y_bounds)
#                    BoxPlotter.plot2DBoxesCombined_temp([boxes_of_interest[i1], boxes_of_interest[i2]], x_bounds, y_bounds, heights)
######## begin 3d demos
    def test_toy_3d(self):
        x = Symbol("x", REAL)
        y = Symbol("y", REAL)
        z = Symbol("z", REAL)
        a1 = Interval.make_interval([1,2])
        a2 = Interval.make_interval([5,6])
        a3 = Interval.make_interval([1.5,2.5])
        x_param = Parameter("x",lb=a1.lb, ub=a1.ub)
        y_param = Parameter("y",y)
        z_param = Parameter("z",z)
        dict = {x_param: a1, y_param: a1, z_param: a2}
        dict2 = {x_param: a3, y_param: a3, z_param: a1}
        print(dict)
        print(dict2)
        print(Box([x_param, lb=a1.lb,ub=a1.ub]))

if __name__ == "__main__":
    unittest.main()
