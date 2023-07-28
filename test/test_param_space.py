import os
import unittest

from funman.representation import ParameterSpace

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
                    "rec_o_o": {
                        "lb": 0.3,
                        "ub": 0.5,
                    },
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


if __name__ == "__main__":
    unittest.main()
