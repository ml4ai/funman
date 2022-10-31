import sys
from funman.search import BoxSearch, SearchConfig
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
from funman.model import Parameter, Model
from funman.scenario.parameter_synthesis import ParameterSynthesisScenario


class TestCompilation(unittest.TestCase):
    def test_box_ops():
        # FIXME Drisana
        ps1 = ParameterSpace(true_boxes, false_boxes)
        ps2 = ParameterSpace(true_boxes, false_boxes)

        ParameterSpace.intersect(ps1, ps2)


if __name__ == "__main__":
    unittest.main()
