import unittest

from matplotlib.lines import Line2D

from funman.representation import ModelParameter
from funman.search import Box, Interval


class TestCompilation(unittest.TestCase):
    ######## begin demos
    def test_symm_diff_1(self):
        # Manually make boxes for example.
        dict1 = {
            "x": Interval(lb=1, ub=2),
            "y": Interval(lb=1, ub=2),
            "z": Interval(lb=1, ub=2),
        }
        dict2 = {
            "x": Interval(lb=1.5, ub=2.5),
            "y": Interval(lb=1.5, ub=2.5),
            "z": Interval(lb=1.5, ub=2.5),
        }
        box1 = Box(bounds=dict1)
        box2 = Box(bounds=dict2)
        result = Box.symm_diff(box1, box2)
        print(result)
        assert result

    def test_symm_diff_2(self):
        # Manually make boxes for example.  Should return the original boxes since the constituent boxes are disjoint
        dict1 = {
            "x": Interval(lb=1, ub=2),
            "y": Interval(lb=1, ub=2),
            "z": Interval(lb=1, ub=2),
        }
        dict2 = {
            "x": Interval(lb=1.5, ub=2.5),
            "y": Interval(lb=1.5, ub=2.5),
            "z": Interval(lb=5, ub=6),
        }
        box1 = Box(bounds=dict1)
        box2 = Box(bounds=dict2)
        result = Box.symm_diff(box1, box2)
        print(result)
        assert result


if __name__ == "__main__":
    unittest.main()
