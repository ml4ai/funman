import json
import os
from contextlib import contextmanager
from time import sleep
from timeit import default_timer
from typing import Dict, Tuple, Union

# import matplotlib.pyplot as plt
import pydantic

from funman.api.settings import Settings
from funman.model.generated_models.petrinet import Model as GeneratedPetriNet
from funman.model.generated_models.regnet import Model as GeneratedRegnet
from funman.model.model import _wrap_with_internal_model
from funman.server.query import (
    FunmanResults,
    FunmanWorkRequest,
    FunmanWorkUnit,
)
from funman.server.storage import Storage
from funman.server.worker import FunmanWorker

# from funman_demo.parameter_space_plotter import ParameterSpacePlotter


# RESOURCES = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)), "../resources"
# )

# out_dir = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)), "out", "evaluation"
# )


models = {GeneratedPetriNet, GeneratedRegnet}

# AMR_EXAMPLES_DIR = os.path.join(RESOURCES, "amr")
# AMR_PETRI_DIR = os.path.join(AMR_EXAMPLES_DIR, "petrinet", "amr-examples")
# AMR_REGNET_DIR = os.path.join(AMR_EXAMPLES_DIR, "regnet", "amr-examples")

# SKEMA_PETRI_DIR = os.path.join(AMR_EXAMPLES_DIR, "petrinet", "skema")
# SKEMA_REGNET_DIR = os.path.join(AMR_EXAMPLES_DIR, "regnet", "skema")

# MIRA_PETRI_DIR = os.path.join(AMR_EXAMPLES_DIR, "petrinet", "mira")


# cases = [
#     # S1 base model
#     (
#         os.path.join(MIRA_PETRI_DIR, "models", "eval_scenario1_base.json"),
#         os.path.join(MIRA_PETRI_DIR, "requests", "eval_scenario1_base.json"),
#     ),
#     # S1 base model ps for beta
#     (
#         os.path.join(MIRA_PETRI_DIR, "models", "eval_scenario1_base.json"),
#         os.path.join(
#             MIRA_PETRI_DIR, "requests", "eval_scenario1_base_ps_beta.json"
#         ),
#     ),
#     # S1 1.ii.1
#     (
#         os.path.join(
#             MIRA_PETRI_DIR, "models", "eval_scenario1_1_ii_1_init1.json"
#         ),
#         os.path.join(MIRA_PETRI_DIR, "requests", "eval_scenario1_1_ii_1.json"),
#     ),
#     # S1 2 # has issue with integer overflow due to sympy taylor series
#     (
#         os.path.join(MIRA_PETRI_DIR, "models", "eval_scenario1_1_ii_2.json"),
#         os.path.join(MIRA_PETRI_DIR, "requests", "eval_scenario1_1_ii_2.json"),
#     ),
#     # S1 3, advanced to t=75, parmsynth to separate (non)compliant
#     # (
#     #     os.path.join(MIRA_PETRI_DIR, "models", "eval_scenario1_1_ii_3_t75.json"),
#     #     os.path.join(MIRA_PETRI_DIR, "requests", "eval_scenario1_1_ii_3_t75_ps.json"),
#     # ),
#     # S3 base for CEIMS
#     (
#         os.path.join(MIRA_PETRI_DIR, "models", "eval_scenario3_base.json"),
#         os.path.join(MIRA_PETRI_DIR, "requests", "eval_scenario3_base.json"),
#     ),
# ]

# speedup_cases = [
#     # baseline: no substitution, no mcts, no query simplification
#     # > 10m
#     (
#         os.path.join(MIRA_PETRI_DIR, "models", "eval_scenario1_base.json"),
#         os.path.join(
#             MIRA_PETRI_DIR, "requests", "eval_scenario1_base_baseline.json"
#         ),
#         "Baseline",
#     ),
#     # mcts: no substitution, no query simplification
#     (
#         os.path.join(MIRA_PETRI_DIR, "models", "eval_scenario1_base.json"),
#         os.path.join(
#             MIRA_PETRI_DIR, "requests", "eval_scenario1_base_mcts.json"
#         ),
#         "MCTS",
#     ),
#     # mcts, substitution, no query simplification
#     (
#         os.path.join(MIRA_PETRI_DIR, "models", "eval_scenario1_base.json"),
#         os.path.join(
#             MIRA_PETRI_DIR, "requests", "eval_scenario1_base_substitution.json"
#         ),
#         "MCTS+Sub+Approx",
#     ),
#     # mcts, substitution, query simplification
#     (
#         os.path.join(MIRA_PETRI_DIR, "models", "eval_scenario1_base.json"),
#         os.path.join(MIRA_PETRI_DIR, "requests", "eval_scenario1_base.json"),
#         "MCTS+Sub+Approx+Compile",
#     ),
# ]


class Runner:
    @contextmanager
    def elapsed_timer(self):
        start = default_timer()
        elapser = lambda: default_timer() - start
        try:
            yield elapser
        finally:
            elapser = None

    def run(
        self, model, request, description="", case_out_dir="."
    ) -> FunmanResults:
        results = self.run_test_case(
            (model, request, description), case_out_dir
        )
        return results
        # ParameterSpacePlotter(
        #     results.parameter_space,
        #     plot_points=True,
        #     parameters=["beta", "num_steps"],
        # ).plot(show=False)
        # plt.savefig(f"{case_out_dir}/scenario1_base_ps_beta_space.png")

    def run_test_case(self, case, case_out_dir):
        if not os.path.exists(case_out_dir):
            os.mkdir(case_out_dir)

        self.settings = Settings()
        self.settings.data_path = case_out_dir
        self._storage = Storage()
        self._worker = FunmanWorker(self._storage)
        self._storage.start(self.settings.data_path)
        self._worker.start()

        results = self.run_instance(case, out_dir=case_out_dir)

        self._worker.stop()
        self._storage.stop()

        return results

    def get_model(self, model_file: str):
        for model in models:
            try:
                with open(model_file, "r") as mf:
                    m = _wrap_with_internal_model(model(**json.load(mf)))
                return m
            except Exception as e:
                pass
        raise Exception(f"Could not determine the Model type of {model_file}")

    def run_instance(
        self, case: Tuple[str, Union[str, Dict], str], out_dir="."
    ):
        (model_file, request_file, description) = case

        model = self.get_model(model_file)

        try:
            with open(request_file, "r") as rf:
                request = FunmanWorkRequest(**json.load(rf))
        except TypeError as te:
            # request_file may not be a path, could be a dict
            try:
                request = FunmanWorkRequest.model_validate(request_file)
            except Exception as e:
                raise e

        work_unit: FunmanWorkUnit = self._worker.enqueue_work(
            model=model, request=request
        )
        sleep(2)  # need to sleep until worker has a chance to start working
        while True:
            if self._worker.is_processing_id(work_unit.id):
                results = self._worker.get_results(work_unit.id)
                with open(f"{out_dir}/{work_unit.id}.json", "w") as f:
                    f.write(results.model_dump_json())
                # ParameterSpacePlotter(
                #     results.parameter_space, plot_points=True
                # ).plot(show=False)
                # plt.savefig(f"{out_dir}/{model.__module__}.png")
                # plt.close()
                sleep(2)
            else:
                results = self._worker.get_results(work_unit.id)
                break

        # ParameterSpacePlotter(results.parameter_space, plot_points=True).plot(
        #     show=False
        # )
        # plt.savefig(f"{out_dir}/{model.__module__}.png")
        # plt.close()

        return results


if __name__ == "__main__":
    model = args[1]
    request = args[2]
    out_dir = args[3]

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    Runner().run(model, request, case_out_dir=out_dir)
