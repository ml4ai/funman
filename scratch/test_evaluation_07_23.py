import os
import unittest
from contextlib import contextmanager
from time import sleep
from timeit import default_timer
from typing import Tuple

import matplotlib.pyplot as plt
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
    os.path.dirname(os.path.abspath(__file__)), "out", "evaluation"
)


models = {GeneratedPetrinet, GeneratedRegnet}

AMR_EXAMPLES_DIR = os.path.join(RESOURCES, "amr")
AMR_PETRI_DIR = os.path.join(AMR_EXAMPLES_DIR, "petrinet", "amr-examples")
AMR_REGNET_DIR = os.path.join(AMR_EXAMPLES_DIR, "regnet", "amr-examples")

SKEMA_PETRI_DIR = os.path.join(AMR_EXAMPLES_DIR, "petrinet", "skema")
SKEMA_REGNET_DIR = os.path.join(AMR_EXAMPLES_DIR, "regnet", "skema")

MIRA_PETRI_DIR = os.path.join(AMR_EXAMPLES_DIR, "petrinet", "mira")


cases = [
    # S1 base model
    (
        os.path.join(MIRA_PETRI_DIR, "models", "eval_scenario1_base.json"),
        os.path.join(MIRA_PETRI_DIR, "requests", "eval_scenario1_base.json"),
    ),
    # S1 base model ps for beta
    (
        os.path.join(MIRA_PETRI_DIR, "models", "eval_scenario1_base.json"),
        os.path.join(
            MIRA_PETRI_DIR, "requests", "eval_scenario1_base_ps_beta.json"
        ),
    ),
    # S1 1.ii.1
    (
        os.path.join(
            MIRA_PETRI_DIR, "models", "eval_scenario1_1_ii_1_init1.json"
        ),
        os.path.join(MIRA_PETRI_DIR, "requests", "eval_scenario1_1_ii_1.json"),
    ),
    # S1 2 # has issue with integer overflow due to sympy taylor series
    (
        os.path.join(MIRA_PETRI_DIR, "models", "eval_scenario1_1_ii_2.json"),
        os.path.join(MIRA_PETRI_DIR, "requests", "eval_scenario1_1_ii_2.json"),
    ),
    # S1 3, advanced to t=75, parmsynth to separate (non)compliant
    # (
    #     os.path.join(MIRA_PETRI_DIR, "models", "eval_scenario1_1_ii_3_t75.json"),
    #     os.path.join(MIRA_PETRI_DIR, "requests", "eval_scenario1_1_ii_3_t75_ps.json"),
    # ),
    # S3 base for CEIMS
    (
        os.path.join(MIRA_PETRI_DIR, "models", "eval_scenario3_base.json"),
        os.path.join(MIRA_PETRI_DIR, "requests", "eval_scenario3_base.json"),
    ),
]

speedup_cases = [
    # baseline: no substitution, no mcts, no query simplification
    # > 10m
    (
        os.path.join(MIRA_PETRI_DIR, "models", "eval_scenario1_base.json"),
        os.path.join(
            MIRA_PETRI_DIR, "requests", "eval_scenario1_base_baseline.json"
        ),
        "Baseline",
    ),
    # mcts: no substitution, no query simplification
    (
        os.path.join(MIRA_PETRI_DIR, "models", "eval_scenario1_base.json"),
        os.path.join(
            MIRA_PETRI_DIR, "requests", "eval_scenario1_base_mcts.json"
        ),
        "MCTS",
    ),
    # mcts, substitution, no query simplification
    (
        os.path.join(MIRA_PETRI_DIR, "models", "eval_scenario1_base.json"),
        os.path.join(
            MIRA_PETRI_DIR, "requests", "eval_scenario1_base_substitution.json"
        ),
        "MCTS+Sub+Approx",
    ),
    # mcts, substitution, query simplification
    (
        os.path.join(MIRA_PETRI_DIR, "models", "eval_scenario1_base.json"),
        os.path.join(MIRA_PETRI_DIR, "requests", "eval_scenario1_base.json"),
        "MCTS+Sub+Approx+Compile",
    ),
]


if not os.path.exists(out_dir):
    os.mkdir(out_dir)


class TestModels(unittest.TestCase):
    @contextmanager
    def elapsed_timer(self):
        start = default_timer()
        elapser = lambda: default_timer() - start
        try:
            yield elapser
        finally:
            elapser = None

    def test_scenario1_base_consistency_speedups(self):
        case_out_dir = os.path.join(
            out_dir, "scenario1_base_consistency_speedup"
        )
        time_results = {}
        for case in speedup_cases:
            print(f"Solving Case: {case}")
            with self.elapsed_timer() as t:
                results = self.run_test_case(case, case_out_dir)
                elapsed = t()
                time_results[case[2]] = elapsed
                print(time_results)

    @unittest.skip(reason="tmp")
    def test_scenario1_base_consistency(self):
        case_out_dir = os.path.join(out_dir, "scenario1_base_consistency")
        case = cases[0]
        with self.elapsed_timer() as t:
            results = self.run_test_case(case, case_out_dir)
            elapsed_base_dreal = t()

        tp = results.parameter_space.true_points[0]
        fig, ax = plt.subplots()
        ax = results.plot([tp], variables=["S", "E", "H", "D", "I", "R"])
        plt.savefig(
            os.path.join(case_out_dir, "scenario1_base_consistency_point.png")
        )

    @unittest.skip(reason="tmp")
    def test_scenario1_base_ps_beta(self):
        case_out_dir = os.path.join(out_dir, "scenario1_base_ps_beta")
        case = cases[1]
        results = self.run_test_case(case, case_out_dir)

        ParameterSpacePlotter(
            results.parameter_space,
            plot_points=True,
            parameters=["beta", "num_steps"],
        ).plot(show=False)
        plt.savefig(f"{case_out_dir}/scenario1_base_ps_beta_space.png")

        # results.plot_trajectories("I")
        # plt.savefig(
        #     os.path.join(case_out_dir, "scenario1_base_ps_beta_trajectories.png")
        # )

        # tps = results.parameter_space.true_points
        # fig, ax = plt.subplots()
        # ax = results.plot(tps, variables=["I"])
        # plt.savefig(os.path.join(case_out_dir, "scenario1_base_ps_beta_space.png"))

    @unittest.skip(reason="tmp")
    def test_scenario1_1_ps_cm_epsm(self):
        case_out_dir = os.path.join(out_dir, "scenario1_1_ps_cm_epsm")
        case = cases[2]
        results = self.run_test_case(case, case_out_dir)

        ParameterSpacePlotter(
            results.parameter_space,
            plot_points=True,
            parameters=["c_m", "eps_m", "num_steps"],
        ).plot(show=False)
        plt.savefig(f"{case_out_dir}/scenario1_1_ps_cm_epsm_space.png")

    @unittest.skip(reason="tmp")
    def test_scenario1_2_ps_t0(self):
        case_out_dir = os.path.join(out_dir, "scenario1_2_ps_t0")
        case = cases[3]
        results = self.run_test_case(case, case_out_dir)

        ParameterSpacePlotter(
            results.parameter_space,
            plot_points=True,
            parameters=["t_0", "num_steps"],
        ).plot(show=False)
        plt.savefig(f"{case_out_dir}/scenario1_2_ps_t0_space.png")

    @unittest.skip(reason="tmp")
    def test_scenario1_3_ps_strat_eps(self):
        case_out_dir = os.path.join(out_dir, "scenario1_3_ps_strat_eps")
        case = cases[4]
        results = self.run_test_case(case, case_out_dir)

        ParameterSpacePlotter(
            results.parameter_space,
            plot_points=True,
            parameters=[
                "eps_m_0",
                "eps_m_1",
                "eps_m_2",
                "eps_m_3",
                "num_steps",
            ],
        ).plot(show=False)
        plt.savefig(f"{case_out_dir}/scenario1_3_ps_strat_eps_space.png")

    @unittest.skip(reason="tmp")
    def test_scenario3_base_ps(self):
        case_out_dir = os.path.join(out_dir, "scenario1_3_base_ps")
        case = cases[5]
        results = self.run_test_case(case, case_out_dir)

        ParameterSpacePlotter(
            results.parameter_space,
            plot_points=True,
            parameters=["beta", "gamma", "lambda", "num_steps"],
        ).plot(show=False)
        plt.savefig(f"{case_out_dir}/scenario3_base_ps_space.png")

    def run_test_case(self, case, case_out_dir):
        if not os.path.exists(case_out_dir):
            os.mkdir(case_out_dir)

        self.settings = Settings()
        self.settings.data_path = case_out_dir
        self._storage = Storage()
        self._worker = FunmanWorker(self._storage)
        self._storage.start(self.settings.data_path)
        self._worker.start()

        results = self.run_instance(case)

        self._worker.stop()
        self._storage.stop()

        return results

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

    def run_instance(self, case: Tuple[str, str, str]):
        (model_file, request_file, description) = case

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
                    f.write(results.json())
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
    unittest.main(module="test_evaluation_07_23")
