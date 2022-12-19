import os
import sys
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
from funman.model import Model, Parameter
from funman.scenario.parameter_synthesis import ParameterSynthesisScenario
from funman.search import BoxSearch, SearchConfig


class TestCompilation(unittest.TestCase):
    def test_box_ops():
        # FIXME Drisana
        ps1 = ParameterSpace(true_boxes, false_boxes)
        ps2 = ParameterSpace(true_boxes, false_boxes)

        ParameterSpace.intersect(ps1, ps2)


if __name__ == "__main__":
    unittest.main()
