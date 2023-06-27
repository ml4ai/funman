import json
import os
from typing import Tuple
import unittest
from time import sleep

import matplotlib.pyplot as plt
from funman_demo.parameter_space_plotter import ParameterSpacePlotter

from funman.api.settings import Settings
from funman.funman import FUNMANConfig
from funman.model.generated_models.petrinet import Model as GeneratedPetrinet
from funman.model.generated_models.regnet import Model as GeneratedRegnet
from funman.server.query import FunmanWorkRequest, FunmanWorkUnit
from funman.server.storage import Storage
from funman.server.worker import FunmanWorker
from funman.api.api import _wrap_with_internal_model
import pydantic

RESOURCES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../resources"
)

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


models = {GeneratedPetrinet, GeneratedRegnet}

AMR_EXAMPLES_DIR = os.path.join(RESOURCES, "amr", "petrinet", "amr-examples")

cases = [
    (os.path.join(AMR_EXAMPLES_DIR, "sir.json"), os.path.join(AMR_EXAMPLES_DIR, "sir_request1.json")),
    # (
    #     os.path.join(AMR_EXAMPLES_DIR, "sir.json"),
    #     os.path.join(AMR_EXAMPLES_DIR, "sir_request2.json"),
    # )
]

if not os.path.exists(out_dir):
    os.mkdir(out_dir)


class TestModels(unittest.TestCase):
    def test_models(self):
        self.settings = Settings()
        self.settings.data_path = out_dir
        self._storage = Storage()
        self._worker = FunmanWorker(self._storage)
        self._storage.start(self.settings.data_path)
        self._worker.start()

        for case in cases:
            self.run_instance(case)

        self._worker.stop()
        self._storage.stop()

    def get_model(self, model_file: str):
        for model in models:
            try:
                m = _wrap_with_internal_model(
                    pydantic.parse_file_as(model, model_file)
                )
                return m
            except Exception as e:
                pass
        raise Exception(f"Could not determine the Model type of {model_file}")

    def run_instance(self, case: Tuple[str, str]):
        (model_file, request_file) = case

        model = self.get_model(model_file)
        request = pydantic.parse_file_as(FunmanWorkRequest, request_file)

        work_unit: FunmanWorkUnit = self._worker.enqueue_work(
            model=model, request=request
        )
        sleep(2)  # need to sleep until worker has a chance to start working
        while True:
            if self._worker.is_processing_id(work_unit.id):
                results = self._worker.get_results(work_unit.id)
                ParameterSpacePlotter(
                    results.parameter_space, plot_points=True
                ).plot(show=True)
                plt.savefig(f"{out_dir}/{model.__module__}.png")
                sleep(2)
            else:
                results = self._worker.get_results(work_unit.id)
                break

        assert results

        assert True


if __name__ == "__main__":
    unittest.main()
