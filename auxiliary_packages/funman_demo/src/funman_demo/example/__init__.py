import os
import tempfile

import matplotlib.pyplot as plt
from funman_demo.handlers import RealtimeResultPlotter, ResultCacheWriter

from funman import Funman
from funman.model import Parameter
from funman.scenario.parameter_synthesis import (
    ParameterSynthesisScenario,
    ParameterSynthesisScenarioResult,
)
from funman.search import ResultCombinedHandler

from ..plot import plot_parameter_space


def realtime_plotting_and_caching():
    tmp_dir_path = tempfile.mkdtemp(prefix="funman-")
    print(f"FUNMAN: Writing example realtime output to {tmp_dir_path}")
    gromet_file1 = "chime1"
    parameters = [
        Parameter(name="beta_0", lb=0.0, ub=0.5),
        Parameter(name="beta_1", lb=0.0, ub=0.5),
    ]
    result1: ParameterSynthesisScenarioResult = Funman().solve(
        ParameterSynthesisScenario(
            parameters,
            gromet_file1,
            config={
                "linearize": True,
                "epochs": [(0, 20), (20, 30)],
                "population_size": 1002,
                "infectious_days": 14.0,
            },
        ),
        SearchConfig(
            tolerance=0.1,
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
    plot_parameter_space(result1.parameter_space)
    plt.savefig(os.path.join(tmp_dir_path, "search-final.png"))
