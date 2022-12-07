import sys
import tempfile
from funman.scenario.consistency import ConsistencyScenario
from funman.search import BoxSearch, SearchConfig, SMTCheck
from funman.search_utils import Box, ResultCombinedHandler
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
from funman.scenario.parameter_synthesis import ParameterSynthesisScenario
from funman_demo.handlers import ResultCacheWriter, RealtimeResultPlotter
from model2smtlib.bilayer.translate import (
    Bilayer,
    BilayerEncodingOptions,
    BilayerModel,
    BilayerMeasurement,
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
    def setup(self, duration=10, transmission_reduction=0.05):
        bilayer_json_file = os.path.join(
            DATA, "CHIME_SIR_dynamics_BiLayer.json"
        )
        bilayer = Bilayer.from_json(bilayer_json_file)
        assert bilayer

        measurements = {
            "state": [{"variable": "I"}],
            "observable": [{"observable": "H"}],
            "rate": [{"parameter": "hr"}],
            "Din": [{"variable": 1, "parameter": 1}],
            "Dout": [{"parameter": 1, "observable": 1}],
        }
        hospital_measurements = BilayerMeasurement.from_json(measurements)

        model = BilayerModel(
            bilayer,
            measurements=hospital_measurements,
            init_values={"S": 10000, "I": 1, "R": 1},
            parameter_bounds={
                "beta": [
                    0.000067,
                    0.000067,
                ],
                # "beta" : [0.00005, 0.00007],
                "gamma": [1.0 / 14.0, 1.0 / 14.0],
                "hr": [0.01, 0.01],
            },
        )

        if isinstance(transmission_reduction, list):
            lb = model.parameter_bounds["beta"][0] * (
                1.0 - transmission_reduction[1]
            )
            ub = model.parameter_bounds["beta"][1] * (
                1.0 - transmission_reduction[0]
            )
        else:
            lb = model.parameter_bounds["beta"][0] * (
                1.0 - transmission_reduction
            )
            ub = model.parameter_bounds["beta"][1] * (
                1.0 - transmission_reduction
            )
        model.parameter_bounds["beta"] = [lb, ub]

        query = QueryLE("H", 0.5)

        encoder = BilayerEncoder(
            config=BilayerEncodingOptions(step_size=1, max_steps=duration)
        )

        return model, query, encoder

    #@unittest.skip("temporarily remove")
    def test_chime_bilayer_solve(self):
        model, query, encoder = self.setup(
            duration=1, transmission_reduction=0.00
        )
        model.init_values["S"] = 100000
        query.ub = 30000

        scenario = ConsistencyScenario(model, query, smt_encoder=encoder)

        result = Funman().solve(
            scenario, config=SearchConfig(solver="dreal", search=SMTCheck)
        )
        assert result

        result.plot(logy=True)
        print(result.dataframe())

    @unittest.skip("temporarily remove")
    def test_chime_bilayer_synthesize(self):

        model, query, encoder = self.setup(
            duration=10, transmission_reduction=[-0.05, 0.15]
        )

        # The efficacy can be up to 4x that of baseline (i.e., 0.05 - 0.20)
        parameters = [
            Parameter(
                "beta",
                # lb=0.000001,
                # ub=0.00001,
                lb=model.parameter_bounds["beta"][0],
                ub=model.parameter_bounds["beta"][1],
            )
        ]
        tmp_dir_path = tempfile.mkdtemp(prefix="funman-")
        result = Funman().solve(
            ParameterSynthesisScenario(
                parameters, model, query, smt_encoder=encoder
            ),
            config=SearchConfig(
                number_of_processes=1,
                tolerance=1e-6,
                solver="dreal",
                search=BoxSearch,
                handler=ResultCombinedHandler(
                    [
                        ResultCacheWriter(
                            os.path.join(tmp_dir_path, "search.json")
                        ),
                        RealtimeResultPlotter(
                            parameters,
                            plot_points=True,
                            realtime_save_path=os.path.join(
                                tmp_dir_path, "search.png"
                            ),
                        ),
                    ]
                ),
            ),
        )
        assert result



if __name__ == "__main__":
    unittest.main()
