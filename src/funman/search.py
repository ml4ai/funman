from functools import total_ordering
from inspect import Parameter
from typing import Dict, List, Union
import time

from funman.scenario import ParameterSynthesisScenario, ParameterSynthesisScenarioResult
from funman.constants import NEG_INFINITY, POS_INFINITY, BIG_NUMBER

from pysmt.shortcuts import get_model, And, LT, LE, GE, TRUE, Not, Real
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from IPython.display import clear_output
from datetime import datetime
from multiprocessing import Pool, cpu_count, Queue, Process, active_children
from queue import Empty

import logging

l = logging.getLogger(__file__)
l.setLevel(logging.ERROR)


class Interval(object):
    def __init__(self, lb: float, ub: float) -> None:
        self.lb = lb
        self.ub = ub
        self.cached_width = None

    def width(self):
        if self.cached_width is None:
            if self.lb == NEG_INFINITY or self.ub == POS_INFINITY:
                self.cached_width = BIG_NUMBER
            else:
                self.cached_width = self.ub - self.lb
        return self.cached_width

    def __lt__(self, other):
        if isinstance(other, Interval):
            return self.width() < other.width()
        else:
            raise Exception(f"Cannot compare __lt__() Interval to {type(other)}")

    def __eq__(self, other):
        if isinstance(other, Interval):
            return self.lb == other.lb and self.ub == other.ub
        else:
            return False

    def __repr__(self):
        return f"[{self.lb}, {self.ub}]"

    def __str__(self):
        return self.__repr__()

    def finite(self) -> bool:
        return self.lb != NEG_INFINITY and self.ub != POS_INFINITY

    def contains(self, other: "Interval") -> bool:
        lhs = (
            (self.lb == NEG_INFINITY or other.lb != NEG_INFINITY)
            if (self.lb == NEG_INFINITY or other.lb == NEG_INFINITY)
            else self.lb <= other.lb
        )
        rhs = (
            (other.ub != POS_INFINITY or self.ub == POS_INFINITY)
            if (self.ub == POS_INFINITY or other.ub == POS_INFINITY)
            else other.ub <= self.ub
        )
        return lhs and rhs

    def intersects(self, other: "Interval") -> bool:
        lhs = (
            (self.lb == NEG_INFINITY or other.lb != NEG_INFINITY)
            if (self.lb == NEG_INFINITY or other.lb == NEG_INFINITY)
            else self.lb <= other.lb
        )
        rhs = (
            (other.ub != POS_INFINITY or self.ub == POS_INFINITY)
            if (self.ub == POS_INFINITY or other.ub == POS_INFINITY)
            else other.ub <= self.ub
        )
        return lhs or rhs

    def midpoint(self):
        if self.lb == NEG_INFINITY and self.ub == POS_INFINITY:
            return 0
        elif self.lb == NEG_INFINITY:
            return self.ub - BIG_NUMBER
        if self.ub == POS_INFINITY:
            return self.lb + BIG_NUMBER
        else:
            return ((self.ub - self.lb) / 2) + self.lb

    def to_smt(self, p: Parameter):
        return And(
            (GE(p.symbol, Real(self.lb)) if self.lb != NEG_INFINITY else TRUE()),
            (LT(p.symbol, Real(self.ub)) if self.ub != POS_INFINITY else TRUE()),
        ).simplify()


@total_ordering
class Box(object):
    def __init__(self, parameters) -> None:
        self.bounds = {p: Interval(NEG_INFINITY, POS_INFINITY) for p in parameters}
        self.cached_width = None

    def to_smt(self):
        return And([interval.to_smt(p) for p, interval in self.bounds.items()])

    def _copy(self):
        c = Box(list(self.bounds.keys()))
        for p, b in self.bounds.items():
            c.bounds[p] = Interval(b.lb, b.ub)
        return c

    def __lt__(self, other):
        if isinstance(other, Box):
            return self.width() > other.width()
        else:
            raise Exception(f"Cannot compare __lt__() Box to {type(other)}")

    def __eq__(self, other):
        if isinstance(other, Box):
            return all([self.bounds[p] == other.bounds[p] for p in self.bounds.keys()])
        else:
            return False

    def __repr__(self):
        return f"{self.bounds}"

    def __str__(self):
        return self.__repr__()

    def finite(self) -> bool:
        return all([i.finite() for _, i in self.bounds.items()])

    def contains(self, other: "Box") -> bool:
        return all(
            [interval.contains(other.bounds[p]) for p, interval in self.bounds.items()]
        )

    def intersects(self, other: "Box") -> bool:
        return all(
            [
                interval.intersects(other.bounds[p])
                for p, interval in self.bounds.items()
            ]
        )

    def _get_max_width_parameter(self):
        widths = [bounds.width() for _, bounds in self.bounds.items()]
        max_width = max(widths)
        param = list(self.bounds.keys())[widths.index(max_width)]
        return param, max_width

    def width(self) -> float:
        if self.cached_width is None:
            _, width = self._get_max_width_parameter()
            self.cached_width = width

        return self.cached_width

    def split(self):
        p, _ = self._get_max_width_parameter()
        mid = self.bounds[p].midpoint()

        b1 = self._copy()
        b2 = self._copy()

        # b1 is lower half
        b1.bounds[p] = Interval(b1.bounds[p].lb, mid)

        # b2 is upper half
        b2.bounds[p] = Interval(mid, b2.bounds[p].ub)

        return [b1, b2]


class SearchStatistics(object):
    def __init__(self):
        self.num_true = 0
        self.num_false = 0
        self.num_unknown = 0
        self.residual = []
        self.last_time = None
        self.iteration_time = []
        self.iteration_operation = []


class SearchConfig(object):
    def __init__(self, *args, **kwargs) -> None:
        self.tolerance = kwargs["tolerance"] if "tolerance" in kwargs else 1e-1
        self.queue_timeout = 1


class SearchEpisode(object):
    def __init__(self) -> None:
        self.statistics = SearchStatistics()


class BoxSearchEpisode(SearchEpisode):
    def __init__(
        self, config: SearchConfig, problem: ParameterSynthesisScenario
    ) -> None:
        super(BoxSearchEpisode, self).__init__()
        self.unknown_boxes = Queue()  # queue.PriorityQueue()
        self.true_boxes = Queue()
        self.false_boxes = Queue()
        self.config = config
        self.problem = problem
        self.num_parameters = len(self.problem.parameters)
        self.iteration = 0

    def initial_box(self) -> Box:
        return Box(self.problem.parameters)

    def on_start(self):
        self.statistics.last_time = datetime.now()

    def on_iteration(self):
        l.debug(f"false boxes: {self.false_boxes}")
        l.debug(f"true boxes: {self.true_boxes}")
        l.debug(f"unknown boxes: {self.unknown_boxes}")
        # if self.iteration % 50 == 0:
        #     self.plot(
        #         self.true_boxes,
        #         self.false_boxes,
        #         self.unknown_boxes,
        #         title=f"Residual = {self.statistics.residual[-1]}, |boxes|={self.statistics.num_unknown}",
        #     )
        self.iteration += 1

    def add_unknown(self, box: Union[Box, List[Box]]):
        if isinstance(box, list):
            for b in box:
                if b.width() > self.config.tolerance:
                    self.unknown_boxes.put(b, timeout=self.config.queue_timeout)
                    self.statistics.num_unknown += 1
        else:
            if box.width() > self.config.tolerance:
                self.unknown_boxes.put(box)
                self.statistics.num_unknown += 1

    def add_false(self, box: Box):
        self.false_boxes.put_nowait(box)
        self.statistics.num_false += 1
        self.statistics.iteration_operation.append("f")

    def add_true(self, box: Box):
        self.true_boxes.put_nowait(box)
        self.statistics.num_true += 1
        self.statistics.iteration_operation.append("t")

    def get_unknown(self):
        box = self.unknown_boxes.get(timeout=self.config.queue_timeout)
        self.statistics.num_unknown -= 1
        self.statistics.residual.append(box.width())
        this_time = datetime.now()
        self.statistics.iteration_time.append(this_time - self.statistics.last_time)
        self.statistics.last_time = this_time
        return box

    def plotBox(self, interval: Interval = Interval(-20, 20)):
        box = Box(self.problem.parameters)
        for p, _ in box.bounds.items():
            box.bounds[p] = interval
        return box

    def plot(
        self,
        true_boxes,
        false_boxes,
        unknown_boxes,
        plot_bounds: Box = None,
        title="Feasible Regions",
    ):
        if not plot_bounds:
            plot_bounds = self.plotBox()

        clear_output(wait=True)

        if self.num_parameters == 1:
            self.plot1D(true_boxes, false_boxes, unknown_boxes, plot_bounds)
        elif self.num_parameters == 2:
            self.plot2D(true_boxes, false_boxes, unknown_boxes, plot_bounds)
        else:
            raise Exception(
                f"Plotting for {self.num_parameters} >= 2 is not supported."
            )

        custom_lines = [
            Line2D([0], [0], color="g", lw=4),
            Line2D([0], [0], color="r", lw=4),
        ]
        plt.title(title)
        plt.legend(custom_lines, ["true", "false"])
        plt.show(block=False)

    def plot1DBox(self, i: Box, p1: Parameter, color="g"):
        x_values = [i.bounds[p1].lb, i.bounds[p1].ub]
        plt.plot(x_values, np.zeros(len(x_values)), color, linestyle="-")

    def plot2DBox(self, i: Box, p1: Parameter, p2: Parameter, color="g"):
        x_limits = i.bounds[p1]
        y_limits = i.bounds[p2]
        x = np.linspace(x_limits.lb, x_limits.ub, 1000)
        plt.fill_between(x, y_limits.lb, y_limits.ub, color=color)

    def plot1D(self, true_boxes, false_boxes, unknown_boxes, plot_bounds):
        p1 = self.problem.parameters[0]
        clear_output(wait=True)
        for i in iter(true_boxes.get, None):
            if plot_bounds.intersects(i) and i.finite():
                self.plot1DBox(i, p1, color="g")

        for i in iter(false_boxes.get, None):
            if plot_bounds.intersects(i) and i.finite():
                self.plot1DBox(i, p1, color="r")
        plt.xlabel(p1.name)

        plt.xlim([plot_bounds.bounds[p1].lb, plot_bounds.bounds[p1].ub])

    def plot2D(self, true_boxes, false_boxes, unknown_boxes, plot_bounds):
        p1 = self.problem.parameters[0]
        p2 = self.problem.parameters[1]

        for i in list(false_boxes.queue):
            if plot_bounds.intersects(i) and i.finite():
                self.plot2DBox(i, p1, p2, color="r")
        for i in list(true_boxes.queue):
            if plot_bounds.intersects(i) and i.finite():
                self.plot2DBox(i, p1, p2, color="g")
        plt.xlabel(p1.name)
        plt.ylabel(p2.name)
        plt.xlim([plot_bounds.bounds[p1].lb, plot_bounds.bounds[p1].ub])
        plt.ylim([plot_bounds.bounds[p2].lb, plot_bounds.bounds[p2].ub])


class BoxSearch(object):
    def __init__(self) -> None:

        self.episodes = []

    def split(self, box: Box, episode: BoxSearchEpisode):
        b1, b2 = box.split()
        episode.add_unknown([b1, b2])
        episode.statistics.iteration_operation.append("s")

    def expand(self, rval: Queue, episode: BoxSearchEpisode):
        while True:
            try:
                box = episode.get_unknown()
            except Empty:
                break
            else:
                # Check whether box intersects f (false region)
                phi = And(box.to_smt(), Not(episode.problem.model.formula))
                res = get_model(phi)
                if res:
                    # box intersects f (false region)

                    # Check whether box intersects t (true region)
                    phi1 = And(box.to_smt(), episode.problem.model.formula)
                    res1 = get_model(phi1)
                    if res1:
                        # box intersects both t and f, so it must be split
                        self.split(box, episode)
                    else:
                        # box is a subset of f (intersects f but not t)
                        episode.add_false(box)  # TODO consider merging lists of boxes

                else:
                    # box does not intersect f, so it is in t (true region)
                    episode.add_true(box)
                episode.on_iteration()
        rval.put(episode.iteration)
        # return True

    def search(
        self, problem: ParameterSynthesisScenario, config: SearchConfig = SearchConfig()
    ) -> Dict[str, List[Box]]:

        episode = BoxSearchEpisode(config, problem)
        self.episodes.append(episode)
        episode.on_start()

        episode.add_unknown(episode.initial_box())
        number_of_processes = cpu_count()
        processes = []
        rval = Queue()

        # creating processes
        for w in range(number_of_processes):
            p = Process(
                target=self.expand,
                args=(
                    rval,
                    episode,
                ),
            )
            processes.append(p)
            p.start()

        results = [rval.get() for p in processes]

        for p in processes:
            p.terminate()

        # completing process
        for p in processes:
            p.join()

        # for p in processes:
        #     r = rval.get()
        #     print(f"Got {r} iterations from process")

        return episode
