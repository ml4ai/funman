import logging
import os
import tempfile
import unittest

import matplotlib.pyplot as plt
import pandas as pd
from funman_demo.handlers import RealtimeResultPlotter, ResultCacheWriter
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
from funman.model import Model, Parameter, QueryLE
from funman.representation.representation import (
    Point,
    ResultCombinedHandler,
    SearchConfig,
)
from funman.scenario.consistency import ConsistencyScenario
from funman.scenario.parameter_synthesis import ParameterSynthesisScenario
from funman.search import BoxSearch, SMTCheck
from funman.translate.bilayer import (
    Bilayer,
    BilayerEncoder,
    BilayerEncodingOptions,
    BilayerMeasurement,
    BilayerModel,
)
from funman.utils.smtlib_utils import smtlibscript_from_formula

l = logging.getLogger(__file__)
l.setLevel(logging.ERROR)

DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../../resources/bilayer"
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

    # @unittest.skip("temporarily remove")
    def test_chime_bilayer_solve(self):
        model, query, encoder = self.setup(
            duration=1, transmission_reduction=0.00
        )

        query.ub = 10000

        scenario = ConsistencyScenario(model, query, _smt_encoder=encoder)

        result = Funman().solve(
            scenario, config=SearchConfig(solver="dreal", search=SMTCheck)
        )
        assert result

        result.plot(logy=True)
        print(result.dataframe())

    # @unittest.skip("temporarily remove")
    def test_chime_bilayer_synthesize(self):

        model, query, encoder = self.setup(
            duration=10, transmission_reduction=[-0.05, 0.1]
        )
        model.init_values = {
            "S": 7438.567991,
            "I": 2261.927694,
            "R": 299.504315,
        }
        query.ub = 75
        # The efficacy can be up to 4x that of baseline (i.e., 0.05 - 0.20)
        parameters = [
            Parameter(
                name="beta",
                # lb=0.000001,
                # ub=0.00001,
                lb=model.parameter_bounds["beta"][0],
                ub=model.parameter_bounds["beta"][1],
            )
        ]
        tmp_dir_path = tempfile.mkdtemp(prefix="funman-")
        result = Funman().solve(
            ParameterSynthesisScenario(
                parameters, model, query, _smt_encoder=encoder
            ),
            config=SearchConfig(
                number_of_processes=1,
                tolerance=1e-8,
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

        # sample points from true boxes and call
        ps = result.parameter_space
        points = []
        for tbox in ps.true_boxes:
            values = {}
            for p, i in tbox.bounds.items():
                param_assignment = (i.lb + i.ub) * 0.5
                values[p.name] = param_assignment
            points.append(Point.from_dict({"values": values}))

        dfs = result.true_point_timeseries(points=points)
        for point, df in zip(points, dfs):
            print("-" * 80)
            print("Parameter assignments:")
            for param, value in point.values.items():
                print(f"    {param.name} = {value}")
            print("-" * 80)
            print(df)
            print("=" * 80)


if __name__ == "__main__":
    unittest.main()
