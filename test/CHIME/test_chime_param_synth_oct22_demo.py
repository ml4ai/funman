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
                [Parameter("beta_0"), Parameter("beta_1")], gromet_file1
            )
        )


if __name__ == "__main__":
    unittest.main()
