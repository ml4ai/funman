import sys
from funman.search import SearchConfig
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
from funman.funman import Funman
from funman.model import Parameter, Model
from funman.scenario import ParameterSynthesisScenario
from funman.examples.chime import CHIME


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
                Equals(gamma, Real(0.071428571)),
            ]
        )

        model = Model(And(params, init, dynamics, bounds, query))

        scenario = ParameterSynthesisScenario(parameters, model)
        funman = Funman()
        config = SearchConfig(tolerance=1e-1)
        result = funman.solve(scenario, config=config)
        assert result


if __name__ == "__main__":
    unittest.main()
