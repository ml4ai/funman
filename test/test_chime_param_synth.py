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
from funman.examples.chime import CHIME

import logging

l = logging.getLogger(__file__)
l.setLevel(logging.ERROR)


class TestCompilation(unittest.TestCase):
    def test_chime(self):

        num_timepoints = 5
        (
            susceptible,
            infected,
            recovered,
            susceptible_n,
            infected_n,
            recovered_n,
            scale,
            beta,
            gamma,
            n,
        ) = CHIME.make_chime_variables(num_timepoints)
        _, init, dynamics, bounds = CHIME.make_chime_model(
            susceptible,
            infected,
            recovered,
            susceptible_n,
            infected_n,
            recovered_n,
            scale,
            beta,
            gamma,
            n,
            num_timepoints,
        )
        query = CHIME.make_chime_query(infected, num_timepoints)

        parameters = [Parameter("beta", beta)]

        params = And(
            [
                # Equals(beta, Real(6.7857e-05)),
                # LE(beta, Real(1e-4)),
                # GE(beta, Real(1e-6)),
                Equals(gamma, Real(0.071428571)),
            ]
        )

        model = Model(And(params, init, dynamics, bounds))

        scenario = ParameterSynthesisScenario(parameters, model, BoxSearch())
        funman = Funman()
        config = SearchConfig(tolerance=1e-1, queue_timeout=10)
        parameter_space = funman.solve(scenario, config=config)
        l.info(f"True Boxes: {parameter_space.true_boxes}")
        assert parameter_space


if __name__ == "__main__":
    unittest.main()
