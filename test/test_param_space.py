import os
import unittest

from funman.search.representation import ParameterSpace

RESOURCES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../resources"
)


class TestCompilation(unittest.TestCase):
    def test_1d(self):
        param_space_cache_file = os.path.join(
            RESOURCES, "cached", "param_space_1d.json"
        )
        ps = ParameterSpace.from_file(param_space_cache_file)
        assert len(ps.true_boxes) == 2
        assert len(ps.false_boxes) == 6
        assert ps.consistent()

        ps._compact()
        assert len(ps.true_boxes) == 2
        assert len(ps.false_boxes) == 2
        assert ps.consistent()


if __name__ == "__main__":
    unittest.main()
