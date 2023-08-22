import os
import unittest
from time import sleep
from typing import Tuple

import matplotlib.pyplot as plt
import pydantic
from parameterized import parameterized

from funman.utils.run import Runner

from funman_demo.parameter_space_plotter import ParameterSpacePlotter

from funman.api.api import _wrap_with_internal_model
from funman.api.settings import Settings
from funman.funman import FUNMANConfig
from funman.model.generated_models.petrinet import Model as GeneratedPetrinet
from funman.model.generated_models.regnet import Model as GeneratedRegnet
from funman.server.query import FunmanWorkRequest, FunmanWorkUnit
from funman.server.storage import Storage
from funman.server.worker import FunmanWorker

RESOURCES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../resources"
)
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
models = {GeneratedPetrinet, GeneratedRegnet}

AMR_EXAMPLES_DIR = os.path.join(RESOURCES, "amr")
AMR_PETRI_DIR = os.path.join(AMR_EXAMPLES_DIR, "petrinet", "terrarium-tests")

cases = [
    (
        os.path.join(AMR_PETRI_DIR, "t01_request.json"),
        os.path.join(AMR_PETRI_DIR, "t01_model.json"),
        "Nelson's test case using ub query on I and parameter ranges.  Consistency Problem."
    ),
]

if not os.path.exists(out_dir):
    os.mkdir(out_dir)


class TestModels(unittest.TestCase):
    @parameterized.expand(cases)
    def test_request(self, request, model, desc):
        result = Runner().run(model, request, case_out_dir=out_dir)
        assert result


if __name__ == "__main__":
    unittest.main()
