import sys
from funman.scenario.consistency import ConsistencyScenario
from funman.search import BoxSearch, SearchConfig
from funman.search_utils import Box
from funman.util import smtlibscript_from_formula
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
from model2smtlib.bilayer.translate import (
    Bilayer,
    BilayerEncodingOptions,
    BilayerModel,
)

import pandas as pd
import matplotlib.pyplot as plt

import logging

l = logging.getLogger(__file__)
l.setLevel(logging.ERROR)

DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../resources/bilayer"
)


class TestChimeBilayerSolve(unittest.TestCase):
    def test_chime_bilayer_solve(self):
        bilayer_json_file = os.path.join(
            DATA, "CHIME_SIR_dynamics_BiLayer.json"
        )
        bilayer = Bilayer.from_json(bilayer_json_file)
        assert bilayer

        scenario = ConsistencyScenario(
            BilayerModel(
                bilayer,
                init_values={"S": 1000, "I": 1, "R": 1},
                parameter_bounds={
                    "beta": [0.00067, 0.00067],
                    "gamma": [1.0 / 14.0, 1.0 / 14.0],
                },
                encoding_options=BilayerEncodingOptions(
                    step_size=4, max_steps=16
                ),
            )
        )

        result = Funman().solve(scenario)
        assert result


if __name__ == "__main__":
    unittest.main()
