import os
import unittest
from functools import partial
from time import sleep
from typing import Tuple

import pydantic
from funman_benchmarks.benchmark import Benchmark
from interruptingcow import timeout

from funman.api.api import _wrap_with_internal_model
from funman.api.settings import Settings
from funman.model.generated_models.petrinet import Model as GeneratedPetrinet
from funman.model.generated_models.regnet import Model as GeneratedRegnet
from funman.server.query import FunmanWorkRequest, FunmanWorkUnit
from funman.server.storage import Storage
from funman.server.worker import FunmanWorker

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

models = {GeneratedPetrinet, GeneratedRegnet}


RESOURCES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../../../../resources"
)

AMR_EXAMPLES_DIR = os.path.join(RESOURCES, "amr")
AMR_PETRI_DIR = os.path.join(AMR_EXAMPLES_DIR, "petrinet", "amr-examples")
AMR_REGNET_DIR = os.path.join(AMR_EXAMPLES_DIR, "regnet", "amr-examples")

SKEMA_PETRI_DIR = os.path.join(AMR_EXAMPLES_DIR, "petrinet", "skema")
SKEMA_REGNET_DIR = os.path.join(AMR_EXAMPLES_DIR, "regnet", "skema")

MIRA_PETRI_DIR = os.path.join(AMR_EXAMPLES_DIR, "petrinet", "mira")


class Evaluation02(Benchmark):


    incremental_speedup_scenarios = [
        # S1 base model, parameter synthesis of beta [0.1, 0.9].  
        # Illustrates the impact of using invariants to evaluate a query incrementally.
        (
            os.path.join(MIRA_PETRI_DIR, "models", "eval_scenario1_base.json"),
            os.path.join(
                MIRA_PETRI_DIR, "requests", "eval_scenario1_base_incremental.json"
            ),
        ),
        # S1 base model, parameter synthesis of beta [0.1, 0.9].  
        # Illustrates the impact of using invariants to evaluate a query non incrementally.
        (
            os.path.join(MIRA_PETRI_DIR, "models", "eval_scenario1_base.json"),
            os.path.join(
                MIRA_PETRI_DIR, "requests", "eval_scenario1_base_non_incremental.json"
            ),
        ),
    ]

    incremental_speedup_cases = [
        {
            "dreal_mcts": dreal_mcts,
            "substitute_subformulas": substitute_subformulas,
            "simplify_query": simplify_query,
            "series_approximation_threshold": series_approximation_threshold,
            "taylor_series_order": taylor_series_order,
            # "profile": True,
        }
        for dreal_mcts in [True]
        for substitute_subformulas in [True]
        for simplify_query in [False]
        for series_approximation_threshold in [1e-10]
        for taylor_series_order in [5]
    ]

    mcts_speedup_scenarios = [
        # S1 base model, parameter synthesis of beta [0.1, 0.9].  
        # Illustrates the impact of using invariants to evaluate a query incrementally.
       (
            os.path.join(MIRA_PETRI_DIR, "models", "eval_scenario1_base.json"),
            os.path.join(
                MIRA_PETRI_DIR, "requests", "eval_scenario1_base_mcts_2param.json"
            ),
        ),
        (
            os.path.join(MIRA_PETRI_DIR, "models", "eval_scenario1_base.json"),
            os.path.join(
                MIRA_PETRI_DIR, "requests", "eval_scenario1_base_mcts.json"
            ),
        ),
         
    ]

    mcts_speedup_cases = [
        {
            "dreal_mcts": dreal_mcts,
            "substitute_subformulas": substitute_subformulas,
            "dreal_precision": dreal_precision,
            "series_approximation_threshold": series_approximation_threshold
        }
        for dreal_mcts in [True, False]
        for dreal_precision in [1e2, 1e1, 1, 1e-1, 1e-3, 1e-5, 1e-10]
        for series_approximation_threshold in [1e-10]
        for substitute_subformulas in [False]
    ]

    scenarios = [
        # S1 base model, parameter synthesis of beta [0.1, 0.9].  
        # Illustrates the impact of using invariants to evaluate a query incrementally.
        (
            os.path.join(MIRA_PETRI_DIR, "models", "eval_scenario1_base.json"),
            os.path.join(
                MIRA_PETRI_DIR, "requests", "eval_scenario1_base_incremental.json"
            ),
        ),
        # S1 base model, parameter synthesis of beta [0.1, 0.9].  
        # Illustrates the impact of using invariants to evaluate a query non incrementally.
        (
            os.path.join(MIRA_PETRI_DIR, "models", "eval_scenario1_base.json"),
            os.path.join(
                MIRA_PETRI_DIR, "requests", "eval_scenario1_base_non_incremental.json"
            ),
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
            os.path.join(
                MIRA_PETRI_DIR, "requests", "eval_scenario1_1_ii_1.json"
            ),
        ),
        # S1 2 # has issue with integer overflow due to sympy taylor series
        (
            os.path.join(
                MIRA_PETRI_DIR, "models", "eval_scenario1_1_ii_2.json"
            ),
            os.path.join(
                MIRA_PETRI_DIR, "requests", "eval_scenario1_1_ii_2.json"
            ),
        ),
        # S1 3, advanced to t=75, parmsynth to separate (non)compliant
        # (
        #     os.path.join(MIRA_PETRI_DIR, "models", "eval_scenario1_1_ii_3_t75.json"),
        #     os.path.join(MIRA_PETRI_DIR, "requests", "eval_scenario1_1_ii_3_t75_ps.json"),
        # ),
        # S3 base for CEIMS
        (
            os.path.join(MIRA_PETRI_DIR, "models", "eval_scenario3_base.json"),
            os.path.join(
                MIRA_PETRI_DIR, "requests", "eval_scenario3_base.json"
            ),
        ),
    ]

    cases = [
        {
            "dreal_mcts": dreal_mcts,
            "substitute_subformulas": substitute_subformulas,
            "simplify_query": simplify_query,
            "series_approximation_threshold": series_approximation_threshold,
            "taylor_series_order": taylor_series_order,
            "profile": True,
        }
        for dreal_mcts in [True]
        for substitute_subformulas in [True]
        for simplify_query in [True]
        for series_approximation_threshold in [1e-5]
        for taylor_series_order in [5]
    ]

    # cases = [{
    #     "dreal_mcts": dreal_mcts,
    #     "substitute_subformulas": substitute_subformulas,
    #     "simplify_query": simplify_query,
    #     "series_approximation_threshold": series_approximation_threshold,
    #      "taylor_series_order": taylor_series_order
    #      } for dreal_mcts in [True, False] for substitute_subformulas in [True, False] for simplify_query in [True, False] for series_approximation_threshold in [1e-1, 1e-3, 1e-5] for taylor_series_order in [1, 3, 5]]

    out_file = os.path.join(out_dir, "results_temp2.json")
    test_timeout = 600

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

    @unittest.skip(reason="tmp")
    def test_model_s1_base_incremental(self):
        scenario_out_dir = os.path.join(out_dir, "scenario1_base_incremental")
        scenarios = self.incremental_speedup_scenarios
        run_case_fn = partial(self.run_test_case, scenarios[0], scenario_out_dir)
        self.run_cases(run_case_fn, self.incremental_speedup_cases, scenarios[0])
        run_case_fn = partial(self.run_test_case, scenarios[1], scenario_out_dir)
        self.run_cases(run_case_fn, self.incremental_speedup_cases, scenarios[1])

    # @unittest.skip(reason="tmp")
    def test_model_s1_base_mcts(self):
        scenario_out_dir = os.path.join(out_dir, "scenario1_base_mcts")
        scenarios = self.mcts_speedup_scenarios
        run_case_fn = partial(self.run_test_case, scenarios[0], scenario_out_dir)
        self.run_cases(run_case_fn, self.mcts_speedup_cases, scenarios[0])
        


    def run_test_case(self, scenario, scenario_out_dir, options):
        if not os.path.exists(scenario_out_dir):
            os.mkdir(scenario_out_dir)

        self.settings = Settings()
        self.settings.data_path = scenario_out_dir
        self._storage = Storage()
        self._worker = FunmanWorker(self._storage)
        self._storage.start(self.settings.data_path)
        self._worker.start()

        results = self.run_instance(scenario, options)

        self._worker.stop()
        self._storage.stop()

        return results

    def run_instance(self, case: Tuple[str, str], options):
        (model_file, request_file) = case

        model = self.get_model(model_file)
        request = pydantic.parse_file_as(FunmanWorkRequest, request_file)
        request.config.__dict__.update(options)
        work_unit: FunmanWorkUnit = self._worker.enqueue_work(
            model=model, request=request
        )
        sleep(1)  # need to sleep until worker has a chance to start working
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
                sleep(1)
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
    unittest.main()
