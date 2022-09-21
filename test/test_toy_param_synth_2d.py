from pysmt.shortcuts import get_model, And, Symbol, FunctionType, Function, Equals, Int, Real, substitute, TRUE, FALSE, Iff, Plus, ForAll, LT, simplify, GT, LE, GE
import sys
from pysmt.typing import INT, REAL, BOOL
import unittest
import os
from funman.funman import Funman, ParameterSynthesisScenario, Parameter, Model

class TestCompilation(unittest.TestCase):
    def test_toy(self):
        
        x = Symbol("x", REAL)
        y = Symbol("y", REAL)
        parameters = [Parameter(x), Parameter(y)]

        # 0.0 < x < 5.0, 10.0 < y < 12.0
        model = Model(
            And(
                LE(x, Real(5.0)), 
                GE(x, Real(0.0)),
                LE(y, Real(12.0)), 
                GE(y, Real(10.0))
                )
            )
        
        scenario = ParameterSynthesisScenario(parameters, model)
        funman = Funman()
        result = funman.solve(scenario)


if __name__ == '__main__':
    unittest.main()
