import os
import unittest

from funman.funman import Funman
from funman.model import Parameter, Model
from funman.scenario import ParameterSynthesisScenario
from funman.scenario import ParameterSynthesisScenarioResult

RESOURCES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "resources")
CACHED = os.path.join(RESOURCES, "cached")

class TestChimeSynth(unittest.TestCase):
    def test_chime(self):
        gromet_file1 = "chime1"
        result1: ParameterSynthesisScenarioResult = Funman().solve(
            ParameterSynthesisScenario(
                [Parameter("beta_0", lb=0.0, ub=0.5), Parameter("beta_1", lb=0.0, ub=0.5)],
                gromet_file1,
                config={
                    "epochs": [(0, 20), (20, 30)],
                    "population_size": 1002,
                    "infectious_days": 14.0,
                    # "read_cache_parameter_space" : os.path.join(CACHED, "parameter_space_2.json"),
                    # "read_cache_parameter_space" : os.path.join(CACHED, "result1.json"),
                    "write_cache_parameter_space" : os.path.join(CACHED, "result2.json"),
                    "real_time_plotting": True
                    # "population_size": 10002,
                    # "infectious_days": 7.0,
                },
            )
        )
        result1.parameter_space.plot()
        pass


if __name__ == "__main__":
    unittest.main()
