import json
import os
import unittest
from time import sleep

import matplotlib.pyplot as plt
from funman_demo.parameter_space_plotter import ParameterSpacePlotter

from funman.api.settings import Settings
from funman.funman import FUNMANConfig
from funman.model.generated_models.petrinet import Model as GeneratedPetrinet
from funman.model.generated_models.regnet import Model as GeneratedRegnet
from funman.model.petrinet import GeneratedPetriNetModel
from funman.model.query import QueryLE, QueryTrue
from funman.model.regnet import GeneratedRegnetModel
from funman.server.query import FunmanWorkRequest, FunmanWorkUnit
from funman.server.storage import Storage
from funman.server.worker import FunmanWorker

RESOURCES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../resources"
)

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


models = {
    os.path.join(
        RESOURCES, "common_model", "petrinet", "sir.json"
    ): GeneratedPetrinet,
    # os.path.join(
    #     RESOURCES, "common_model", "regnet", "lotka_volterra.json"
    # ): GeneratedRegnet,
}

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

        for model in models:
            self.run_instance(model)

        self._worker.stop()
        self._storage.stop()

    def run_instance(self, model_file: str):
        with open(model_file, "r") as f:
            model_json = json.load(f)
        generated_model = models[model_file].parse_raw(json.dumps(model_json))

        config = FUNMANConfig(
            # solver="dreal",
            # dreal_mcts=True,
            tolerance=1e-8,
            number_of_processes=1,
            # save_smtlib=True,
            # dreal_log_level="debug",
        )

        if isinstance(generated_model, GeneratedRegnet):
            model = GeneratedRegnetModel(
                regnet=generated_model,
                structural_parameter_bounds={
                    "num_steps": [2, 2],
                    "step_size": [1, 1],
                },
            )
            parameters = [
                
            ]
            query = QueryLE(variable="W", ub=10)

        elif isinstance(generated_model, GeneratedPetrinet):
            model = GeneratedPetriNetModel(petrinet=generated_model)
            parameters = [
                {"name": "S", "lb": 1, "ub": 10, "label": "all"}
            ]
            query = QueryTrue()

        request = FunmanWorkRequest(
            query=query, parameters=parameters, config=config
        )

        # scenario = ParameterSynthesisScenario(
        #     parameters=parameters,
        #     model=model,
        #     query=query,
        # )

        work_unit: FunmanWorkUnit = self._worker.enqueue_work(
            model=model, request=request
        )
        sleep(2)  # need to sleep until worker has a chance to start working
        while True:
            if self._worker.is_processing_id(work_unit.id):
                sleep(1)
            else:
                results = self._worker.get_results(work_unit.id)
                break

        assert results

        ParameterSpacePlotter(results.parameter_space, plot_points=True).plot(
            show=True
        )
        plt.savefig(f"{out_dir}/{model.name}.png")

        assert True

        # result: ParameterSynthesisScenarioResult = Funman().solve(
        #     scenario,
        #     config=FUNMANConfig(
        #         # solver="dreal",
        #         # dreal_mcts=True,
        #         tolerance=1e-8,
        #         number_of_processes=1,
        #         save_smtlib=True,
        #         # dreal_log_level="debug",
        #         _handler=ResultCombinedHandler(
        #             [
        #                 ResultCacheWriter(f"box_search.json"),
        #                 RealtimeResultPlotter(
        #                     scenario.parameters,
        #                     plot_points=True,
        #                     title=f"Feasible Regions (beta)",
        #                     realtime_save_path=f"box_search.png",
        #                 ),
        #             ]
        #         ),
        #     ),
        # )
        # assert result


if __name__ == "__main__":
    unittest.main()
