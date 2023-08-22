import json
import os
import unittest
from time import sleep
from typing import Tuple

import pydantic
from funman_demo.parameter_space_plotter import ParameterSpacePlotter

from funman.api.api import _wrap_with_internal_model
from funman.api.settings import Settings
from funman.model.generated_models.petrinet import Model as GeneratedPetrinet
from funman.model.generated_models.regnet import Model as GeneratedRegnet
from funman.server.query import FunmanWorkRequest, FunmanWorkUnit
from funman.server.storage import Storage
from funman.server.worker import FunmanWorker

RESOURCES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../resources"
)

out_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "out", "hackathon"
)


models = {GeneratedPetrinet, GeneratedRegnet}

AMR_EXAMPLES_DIR = os.path.join(RESOURCES, "amr")
AMR_PETRI_DIR = os.path.join(AMR_EXAMPLES_DIR, "petrinet", "amr-examples")
AMR_REGNET_DIR = os.path.join(AMR_EXAMPLES_DIR, "regnet", "amr-examples")

SKEMA_PETRI_DIR = os.path.join(AMR_EXAMPLES_DIR, "petrinet", "skema")
SKEMA_REGNET_DIR = os.path.join(AMR_EXAMPLES_DIR, "regnet", "skema")

MIRA_PETRI_DIR = os.path.join(AMR_EXAMPLES_DIR, "petrinet", "mira")


cases = [
    # Buggy model cannot satisfy compartmental constraints
    (
        os.path.join(
            MIRA_PETRI_DIR, "models", "scenario2_a_beta_scale_static.json"
        ),
        os.path.join(
            MIRA_PETRI_DIR,
            "requests",
            "request2_b_default_w_compartmental_constrs.json",
        ),
        0,
    ),
    # Buggy model works if no compartmental constraints
    (
        os.path.join(
            MIRA_PETRI_DIR, "models", "scenario2_a_beta_scale_static.json"
        ),
        os.path.join(
            MIRA_PETRI_DIR,
            "requests",
            "request2_b_default_wo_compartmental_constrs.json",
        ),
        1,
    ),
    # Fixed model can satisfy compartmental constraints
    (
        os.path.join(
            MIRA_PETRI_DIR,
            "models",
            "scenario2_a_beta_scale_static_fixed.json",
        ),
        os.path.join(
            MIRA_PETRI_DIR,
            "requests",
            "request2_b_default_w_compartmental_constrs.json",
        ),
        1,
    ),
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

    def run_instance(self, case: Tuple[str, str, int]):
        (model_file, request_file, expected_num_true_points) = case

        model = self.get_model(model_file)
        request = pydantic.parse_file_as(FunmanWorkRequest, request_file)

        work_unit: FunmanWorkUnit = self._worker.enqueue_work(
            model=model, request=request
        )
        sleep(2)  # need to sleep until worker has a chance to start working
        while True:
            if self._worker.is_processing_id(work_unit.id):
                results = self._worker.get_results(work_unit.id)
                with open(f"{out_dir}/{work_unit.id}.json", "w") as f:
                    f.write(json.dumps(results.json()))
                # ParameterSpacePlotter(
                #     results.parameter_space, plot_points=True
                # ).plot(show=False)
                # plt.savefig(f"{out_dir}/{model.__module__}.png")
                # plt.close()
                sleep(2)
            else:
                results = self._worker.get_results(work_unit.id)
                break

        assert (
            len(results.parameter_space.true_points)
            == expected_num_true_points
        )


if __name__ == "__main__":
    unittest.main()
