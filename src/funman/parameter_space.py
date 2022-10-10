from typing import List
from funman.plotting import BoxPlotter
from funman.search_episode import SearchEpisode
from funman.search_utils import Box


class ParameterSpace(object):
    def __init__(self, true_boxes: List[Box], false_boxes: List[Box]) -> None:
        self.true_boxes = true_boxes
        self.false_boxes = false_boxes

    # STUB project parameter space onto a parameter
    @staticmethod
    def project() -> "ParameterSpace":
        raise NotImplementedError()
        return ParameterSpace()

    # STUB intersect parameters spaces
    @staticmethod
    def intersect(ps1, ps2) -> "ParameterSpace":
        # FIXME Drisana
        raise NotImplementedError()
        return ParameterSpace()

    # STUB symmetric_difference of parameter spaces
    @staticmethod
    def symmetric_difference(ps1, ps2) -> "ParameterSpace":
        # FIXME Drisana
        raise NotImplementedError()
        return ParameterSpace()

    # STUB construct space where all parameters are equal
    @staticmethod
    def construct_all_equal(ps) -> "ParameterSpace":
        raise NotImplementedError()
        return ParameterSpace()

    # STUB compare parameter spaces for equality
    @staticmethod
    def compare(ps1, ps2) -> bool:
        raise NotImplementedError()
        raise NotImplementedError()

    def plot():
        # FIXME Drisana
        box_plotter = BoxPlotter([], [])
        box_plotter.new_plot_fn()
        pass
