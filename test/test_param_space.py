import json
import os
import unittest
from pathlib import Path

from pysmt.shortcuts import GE, LE, And, Real, Symbol
from pysmt.typing import REAL

from funman.api.api import _wrap_with_internal_model
from funman.funman import Funman, FUNMANConfig
from funman.model import EncodedModel, QueryTrue
from funman.model.generated_models.petrinet import Model as GeneratedPetriNet
from funman.representation import ParameterSpace
from funman.representation.representation import ModelParameter
from funman.scenario.parameter_synthesis import ParameterSynthesisScenario
from funman.server.query import FunmanWorkUnit
from funman.translate import EncodedEncoder

RESOURCES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../resources"
)

AMR_DIR = Path(RESOURCES) / "amr"
MIRA_PETRI_DIR = AMR_DIR / "petrinet" / "mira"
MIRA_PETRI_MODELS = MIRA_PETRI_DIR / "models"
MIRA_PETRI_REQUESTS = MIRA_PETRI_DIR / "requests"


class TestCompilation(unittest.TestCase):
    def test_1d(self):
        param_space_cache_file = os.path.join(
            RESOURCES, "cached", "param_space_1d.json"
        )
        ps = ParameterSpace(num_dimensions=2)
        ps.append_result(
            {
                "type": "point",
                "label": "true",
                "values": {"inf_o_o": 0.25, "rec_o_o": 0.25},
            }
        )
        assert len(ps.true_points) == 1

        ps.append_result(
            {
                "type": "point",
                "label": "false",
                "values": {"inf_o_o": 0.75, "rec_o_o": 0.75},
            }
        )
        assert len(ps.false_points) == 1

        ps.append_result(
            {
                "type": "box",
                "label": "true",
                "bounds": {
                    "inf_o_o": {"lb": 0, "ub": 0.3},
                    "rec_o_o": {"lb": 0, "ub": 0.3},
                },
            }
        )
        ps.append_result(
            {
                "type": "box",
                "label": "true",
                "bounds": {
                    "inf_o_o": {"lb": 0.0, "ub": 0.3},
                    "rec_o_o": {"lb": 0.3, "ub": 0.5},
                },
            }
        )
        assert len(ps.true_boxes) == 2

        ps.append_result(
            {
                "type": "box",
                "label": "false",
                "bounds": {
                    "inf_o_o": {"lb": 0.5, "ub": 1.0},
                    "rec_o_o": {"lb": 0.5, "ub": 1.0},
                },
            }
        )
        assert len(ps.false_boxes) == 1
        assert ps.consistent()

        ps._compact()
        assert len(ps.true_boxes) == 1
        assert len(ps.false_boxes) == 1
        assert ps.consistent()

    def test_box_volume(self):
        ps = ParameterSpace(num_dimensions=2)

        ps.append_result(
            {
                "type": "box",
                "label": "true",
                "bounds": {
                    "inf_o_o": {"lb": 0.0, "ub": 0.3},
                    "rec_o_o": {"lb": 0.5, "ub": 0.3},
                },
            }
        )

        ps.append_result(
            {
                "type": "box",
                "label": "false",
                "bounds": {
                    "inf_o_o": {"lb": 0.5, "ub": 1.0},
                    "rec_o_o": {"lb": 0.5, "ub": 1.0},
                },
            }
        )

        for box in ps.true_boxes:
            with self.assertRaises(Exception):
                true_volume = box.volume()

        for box in ps.false_boxes:
            false_volume = box.volume()
            assert false_volume == 0.25

    def test_ps_labeled_volume(self):
        ps = ParameterSpace(num_dimensions=2)

        ps.append_result(
            {
                "type": "box",
                "label": "true",
                "bounds": {
                    "inf_o_o": {"lb": 0, "ub": 0.3},
                    "rec_o_o": {"lb": 0, "ub": 0.3},
                },
            }
        )
        ps.append_result(
            {
                "type": "box",
                "label": "true",
                "bounds": {
                    "inf_o_o": {"lb": 0.0, "ub": 0.3},
                    "rec_o_o": {"lb": 0.3, "ub": 0.5},
                },
            }
        )

        ps.append_result(
            {
                "type": "box",
                "label": "false",
                "bounds": {
                    "inf_o_o": {"lb": 0.5, "ub": 1.0},
                    "rec_o_o": {"lb": 0.5, "ub": 1.0},
                },
            }
        )

        ps.append_result(
            {
                "type": "box",
                "label": "unknown",
                "bounds": {
                    "inf_o_o": {"lb": 0.0, "ub": 0.5},
                    "rec_o_o": {"lb": 0.5, "ub": 1.0},
                },
            }
        )

        ps.append_result(
            {
                "type": "box",
                "label": "unknown",
                "bounds": {
                    "inf_o_o": {"lb": 0.3, "ub": 1.0},
                    "rec_o_o": {"lb": 0.0, "ub": 0.5},
                },
            }
        )

        assert ps.labeled_volume() <= 0.4

    def test_search_space_volume(self):
        parameters = [
            ModelParameter(name="x"),
            ModelParameter(name="y"),
        ]
        x = parameters[0].symbol()
        y = parameters[1].symbol()

        # 0.0 < x < 5.0, 10.0 < y < 12.0
        model = EncodedModel()
        model.parameters = parameters
        model._formula = And(
            LE(x, Real(5.0)),
            GE(x, Real(0.0)),
            LE(y, Real(12.0)),
            GE(y, Real(10.0)),
        )

        scenario = ParameterSynthesisScenario(
            parameters=parameters,
            model=model,
            query=QueryTrue(),
        )

        assert (
            scenario.search_space_volume()
            == scenario.representable_space_volume()
        )

    def test_ps_largest_true_box(self):
        ps = ParameterSpace(num_dimensions=2)

        ps.append_result(
            {
                "type": "box",
                "label": "true",
                "bounds": {
                    "inf_o_o": {"lb": 0.0, "ub": 0.3},
                    "rec_o_o": {"lb": 0.0, "ub": 0.3},
                },
            }
        )
        ps.append_result(
            {
                "type": "box",
                "label": "true",
                "bounds": {
                    "inf_o_o": {"lb": 0.0, "ub": 0.3},
                    "rec_o_o": {"lb": 0.3, "ub": 0.5},
                },
            }
        )

        ps.append_result(
            {
                "type": "box",
                "label": "false",
                "bounds": {
                    "inf_o_o": {"lb": 0.5, "ub": 1.0},
                    "rec_o_o": {"lb": 0.5, "ub": 1.0},
                },
            }
        )

        ps.append_result(
            {
                "type": "box",
                "label": "unknown",
                "bounds": {
                    "inf_o_o": {"lb": 0.0, "ub": 0.5},
                    "rec_o_o": {"lb": 0.5, "ub": 1.0},
                },
            }
        )

        ps.append_result(
            {
                "type": "box",
                "label": "true",
                "bounds": {
                    "inf_o_o": {"lb": 0.3, "ub": 1.0},
                    "rec_o_o": {"lb": 0.0, "ub": 0.5},
                },
            }
        )

        assert ps.max_true_volume()[0] == 0.5

    def test_volume(self):
        model_path = Path(
            MIRA_PETRI_MODELS / "scenario2_a_beta_scale_static.json"
        )
        request_path = Path(MIRA_PETRI_REQUESTS / "request2_b_synthesize.json")

        model = json.loads(model_path.read_bytes())
        request = json.loads(request_path.read_bytes())

        work = FunmanWorkUnit.parse_obj(
            {
                "id": "mock_work",
                # TODO improve testing experience when loading model files without using api
                "model": _wrap_with_internal_model(
                    GeneratedPetriNet.parse_obj(model)
                ),
                "request": request,
            }
        )
        scenario = work.to_scenario()
        config = (
            FUNMANConfig()
            if work.request.config is None
            else work.request.config
        )

        search_volume = scenario.search_space_volume()

        # TODO find better way to capture errors on other thread
        failed_callback = False
        prev_ratio = 0.0

        def callback(results: ParameterSpace):
            nonlocal failed_callback
            nonlocal prev_ratio
            labeled_volume = results.labeled_volume()
            ratio = float(labeled_volume / search_volume)
            if ratio < prev_ratio:
                failed_callback = True
                raise Exception(
                    f"labeled volume/search volume ratio decreased:  {prev_ratio} -> {ratio}"
                )
            prev_ratio = ratio

            if not (0.0 <= ratio <= 1.0):
                failed_callback = True
                raise Exception(
                    f"labeled volume/search volume ratio out of bounds:  {labeled_volume}/{search_volume} = {ratio}"
                )

        Funman().solve(scenario, config, resultsCallback=callback)
        assert not failed_callback, "volume ratio check filed"


if __name__ == "__main__":
    unittest.main()
