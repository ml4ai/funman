import os
import unittest

from pysmt.shortcuts import GE, LE, And, Real, Symbol
from pysmt.typing import REAL

from funman.representation import ParameterSpace
from funman.funman import FUNMANConfig
from funman.model import EncodedModel, QueryTrue
from funman.representation.representation import ModelParameter
from funman.scenario.parameter_synthesis import ParameterSynthesisScenario
from funman.translate import EncodedEncoder

RESOURCES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../resources"
)


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
                true_volume = box._get_product_of_parameter_widths()

        for box in ps.false_boxes:
            false_volume = box._get_product_of_parameter_widths()
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

        assert ps.labeled_volume() == 0.4

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
            parameters=parameters, model=model, query=QueryTrue(),
        )

        assert scenario.search_space_volume() == 1000000000000.0

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


if __name__ == "__main__":
    unittest.main()
