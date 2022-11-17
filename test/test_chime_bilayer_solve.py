import sys
from funman.scenario.consistency import ConsistencyScenario
from funman.search import BoxSearch, SearchConfig
from funman.search_utils import Box
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

import logging

l = logging.getLogger(__file__)
l.setLevel(logging.ERROR)

DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../resources/bilayer"
)


class TestChimeBilayerSolve(unittest.TestCase):
    def test_chime_bilayer_solve(self):
        step_size = 2
        max_steps = 2
        state_timepoints = range(0, max_steps + 1, step_size)
        transition_timepoints = range(0, max_steps, step_size)

        bilayer_json_file = os.path.join(
            DATA, "CHIME_SIR_dynamics_BiLayer.json"
        )
        bilayer = Bilayer.from_json(bilayer_json_file)
        assert bilayer

        encoding = bilayer.to_smtlib(state_timepoints)
        assert encoding

        init_values = {"S": 1000, "I": 1, "R": 0}
        init = And(
            [
                Equals(node.to_smtlib(0), Real(init_values[node.parameter]))
                for idx, node in bilayer.state.items()
            ]
        )

        parameter_bounds = {
            "beta": [6.7e-05, 6.7e-05],
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
        parameter_constraints = parameter_box.to_smt()

        model = Model(And(init, parameter_constraints, encoding))

        scenario = ConsistencyScenario(model)
        funman = Funman()
        config = SearchConfig(tolerance=1e-1, queue_timeout=10)
        result = funman.solve(scenario, config=config)
        vars = list(model.formula.get_free_variables())
        vars.sort(key=lambda x: x.symbol_name())
        for var in vars:
            print(f"{var}")
            print(f"{var} = {result.consistent.get_py_value(var)}")
        assert result


if __name__ == "__main__":
    unittest.main()
