import unittest

from funman.funman import Funman
from funman.model import Parameter, Model
from funman.scenario import ParameterSynthesisScenario
from funman.scenario import ParameterSynthesisScenarioResult


class TestChimeSynth(unittest.TestCase):
    def test_chime(self):
        gromet_file1 = "chime1"
        result1: ParameterSynthesisScenarioResult = Funman().solve(
            ParameterSynthesisScenario(
                [Parameter("beta_0", lb=0.0, ub=0.5), Parameter("beta_1", lb=0.0, ub=0.5)],
                gromet_file1,
                config={
                    "epochs": [(0, 20), (20, 40)],
                    "population_size": 1002,
                    "infectious_days": 14.0,
                    # "population_size": 10002,
                    # "infectious_days": 7.0,
                },
            )
        )


if __name__ == "__main__":
    unittest.main()
