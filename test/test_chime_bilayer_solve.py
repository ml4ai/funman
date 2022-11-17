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
from funman.scenario.parameter_synthesis import ParameterSynthesisScenario
from funman.examples.chime import CHIME
from model2smtlib.bilayer.translate import Bilayer

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
        step_size = 3
        max_steps = 30
        state_timepoints = range(0, max_steps + 1, step_size)
        transition_timepoints = range(0, max_steps, step_size)

        out_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "out"
        )
        # print(out_dir)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        smt2_flat_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            out_dir,
            f"chime_bilayer_flat_{step_size}_{max_steps}.smt2",
        )

        bilayer_json_file = os.path.join(
            DATA, "CHIME_SIR_dynamics_BiLayer.json"
        )
        bilayer = Bilayer.from_json(bilayer_json_file)
        assert bilayer

        encoding = bilayer.to_smtlib(state_timepoints)
        assert encoding

        init_values = {"S": 1000, "I": 1, "R": 1}
        init = And(
            [
                Equals(node.to_smtlib(0), Real(init_values[node.parameter]))
                for idx, node in bilayer.state.items()
            ]
        )

        parameter_bounds = {
            "beta": [0.00067, 0.00067],
            "gamma": [1.0 / 14.0, 1.0 / 14.0],
        }
        parameters = [
            Parameter(
                node.parameter,
                lb=parameter_bounds[node.parameter][0],
                ub=parameter_bounds[node.parameter][1],
            )
            for _, node in bilayer.flux.items()
        ]
        timed_parameters = [
            p.timed_copy(timepoint)
            for p in parameters
            for timepoint in transition_timepoints
        ]
        parameter_box = Box(timed_parameters)
        parameter_constraints = parameter_box.to_smt(closed_upper_bound=True)

        model = Model(And(init, parameter_constraints, encoding))

        with open(smt2_flat_file, "w") as f:
            smtlibscript_from_formula(model.formula).serialize(f, daggify=False)

        scenario = ConsistencyScenario(model)
        funman = Funman()
        config = SearchConfig(tolerance=1e-1, queue_timeout=10)
        result = funman.solve(scenario, config=config)
        timeseries = result.timeseries()
        # print(timeseries)

        df = pd.DataFrame.from_dict(timeseries)
        df.interpolate(method="linear").plot(marker="o")
        plt.show(block=False)
        pass


if __name__ == "__main__":
    unittest.main()
