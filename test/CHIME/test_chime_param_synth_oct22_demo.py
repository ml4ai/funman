import os
import tempfile
import unittest

from funman import Funman
from funman.model import Parameter, Model, QueryLE
from funman.model.chime import ChimeModel
from funman.scenario.parameter_synthesis import ParameterSynthesisScenario
from funman.scenario.parameter_synthesis import ParameterSynthesisScenarioResult
from funman.examples.chime import CHIME
from funman.search_utils import ResultCombinedHandler, SearchConfig
from funman_demo.handlers import ResultCacheWriter

from model2smtlib.chime.translate import ChimeEncoder

RESOURCES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "resources"
)
CACHED = os.path.join(RESOURCES, "cached")


class TestChimeSynth(unittest.TestCase):
    @unittest.skip("Time consuming ...")
    def test_chime(self):
        model = ChimeModel(
            init_values={"s": 1000, "i": 1, "r": 1},
            parameter_bounds={
                "beta": [0.00067, 0.00067],
                "gamma": [1.0 / 14.0, 1.0 / 14.0],
            },
            config={
                "epochs": [(0, 20), (20, 30)],
                "population_size": 1002,
                "infectious_days": 14.0,
                # "read_cache_parameter_space" : os.path.join(CACHED, "parameter_space_2.json"),
                # "read_cache_parameter_space" : os.path.join(CACHED, "result1.json"),
                "write_cache_parameter_space": os.path.join(
                    CACHED, "result2.json"
                ),
                "real_time_plotting": True
                # "population_size": 10002,
                # "infectious_days": 7.0,
            },
            chime=CHIME(),
        )
        query = QueryLE("i", 100)
        tmp_dir_path = tempfile.mkdtemp(prefix="funman-")

        result1: ParameterSynthesisScenarioResult = Funman().solve(
            ParameterSynthesisScenario(
                [
                    Parameter("beta_0", lb=0.0, ub=0.5),
                    Parameter("beta_1", lb=0.0, ub=0.5),
                ],
                model,
                query,
                smt_encoder=ChimeEncoder(),
            ),
            config=SearchConfig(
                number_of_processes=1,
                tolerance=1e-6,
                # wait_action = NotebookImageRefresher(os.path.join(tmp_dir_path, "search.png"), sleep_for=1),
                handler=ResultCacheWriter(
                    os.path.join(tmp_dir_path, "search.json")
                ),
            ),
        )
        result1.parameter_space.plot()
        pass


if __name__ == "__main__":
    unittest.main()
