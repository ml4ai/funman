import os
import unittest

import matplotlib.pyplot as plt
from funman_demo.parameter_space_plotter import ParameterSpacePlotter

from funman.server.query import FunmanResults

RESOURCES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../resources"
)

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

jobs = [
   "068cef77-839d-49be-9191-3502ee6d5240"
        ]


if not os.path.exists(out_dir):
    os.mkdir(out_dir)


class TestPlotParameterSpace(unittest.TestCase):
    def test_plot(self):
        for job in jobs:
            results_file = os.path.join(RESOURCES, "cached", f"{job}.json")
            results: FunmanResults = FunmanResults.parse_file(results_file)
            ParameterSpacePlotter(
                results.parameter_space, plot_points=True
            ).plot(show=False)
            plt.savefig(f"{out_dir}/{results.id}.png")
            plt.close()


if __name__ == "__main__":
    unittest.main()
