import sys
from funman.scenario.consistency import ConsistencyScenario
from funman.search import BoxSearch, SearchConfig, SMTCheck
from funman.search_utils import Box
from model2smtlib.bilayer.translate import (
    BilayerEncoder,
    BilayerEncodingOptions,
)
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
from funman.model import Parameter, Model, QueryLE
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

        transmission_reduction = 0.05
        model = BilayerModel(
            bilayer,
            init_values={"S": 10000, "I": 1, "R": 1},
            parameter_bounds={
                "beta": [
                    0.0,
                    1.0
                    # 0.00067 * (1.0 - transmission_reduction),
                    # 0.00067 * (1.0 - transmission_reduction),
                ],
                # "beta" : [0.00005, 0.00007],
                "gamma": [
                    0.0,
                    1.0
                    # 1.0 / 14.0, 1.0 / 14.0
                ],
                # "hr": [0.01, 0.01]
            },
        )

        query = QueryLE("I", 10000)

        duration = 1
        scenario = ConsistencyScenario(
            model,
            query,
            smt_encoder=BilayerEncoder(
                config=BilayerEncodingOptions(step_size=1, max_steps=duration)
            ),  # four months,
        )

        result = Funman().solve(
            scenario, config=SearchConfig(solver="dreal", search=SMTCheck)
        )
        assert result

        result.plot(logy=True)


if __name__ == "__main__":
    unittest.main()
