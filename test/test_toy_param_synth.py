import sys
sys.path.append('/Users/dmosaphir/SIFT/Projects/ASKEM/code/funman/src')
from pysmt.shortcuts import get_model, And, Symbol, FunctionType, Function, Equals, Int, Real, substitute, TRUE, FALSE, Iff, Plus, ForAll, LT, simplify, GT, LE, GE
from pysmt.typing import INT, REAL, BOOL
import unittest
import os
from funman import Funman, ParameterSynthesisScenario, Parameter, Model

class TestCompilation(unittest.TestCase):
    def test_toy(self):
        
        x = Symbol("x", REAL)
        parameters = [Parameter(x)]

        # 0.0 <= x <= 5
        model = Model(
            And(
                LE(x, Real(5.0)), 
                GE(x, Real(0.0))
                )
            )
        
        scenario = ParameterSynthesisScenario(parameters, model)
        funman = Funman()
        result = funman.solve(scenario)


if __name__ == '__main__':
    unittest.main()
