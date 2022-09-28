import ctypes
from curses.ascii import EM
from functools import total_ordering
from inspect import Parameter
import os

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
from multiprocessing import cpu_count, Queue, Process, Value, Array
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
        self.num_true = Value("i", 0)
        self.num_false = Value("i", 0)
        self.num_unknown = Value("i", 0)
        self.residuals = Queue()
        self.current_residual = Value("d", 0.0)
        self.last_time = Array(ctypes.c_wchar, "")
        self.iteration_time = Queue()
        self.iteration_operation = Queue()

    def close(self):
        self.residuals.close()
        self.iteration_time.close()
        self.iteration_operation.close()


class SearchConfig(object):
    def __init__(self, *args, **kwargs) -> None:
        self.tolerance = kwargs["tolerance"] if "tolerance" in kwargs else 1e-1
        self.queue_timeout = kwargs["queue_timeout"] if "queue_timeout" in kwargs else 1


class SearchEpisode(object):
    def __init__(self) -> None:
        self.statistics = SearchStatistics()


class BoxSearchEpisode(SearchEpisode):
    def __init__(
        self, config: SearchConfig, problem: ParameterSynthesisScenario
    ) -> None:
        super(BoxSearchEpisode, self).__init__()
        self.unknown_boxes = Queue()  # queue.PriorityQueue()
        # self.true_boxes = Queue()
        # self.false_boxes = Queue()
        self.true_boxes = []
        self.false_boxes = []

        self.config = config
        self.problem = problem
        self.num_parameters = len(self.problem.parameters)
        self.iteration = Value("i", 0)
        self.internal_process_id = 0

        self.closed = False
        # self.t_boxes = None
        # self.f_boxes = None
        # self.u_boxes = None

    def initial_box(self) -> Box:
        return Box(self.problem.parameters)

    def on_start(self):
        self.statistics.last_time.value = str(datetime.now())

    def close(self):
        self.unknown_boxes.close()
        # self.true_boxes.close()
        # self.false_boxes.close()
        self.statistics.close()

    # def read_queues(self):
    #     # self.true_boxes.put_nowait(None)
    #     # self.true_boxes.close()
    #     self.t_boxes = []
    #     try:
    #         while True:
    #             self.t_boxes.append(self.true_boxes.get_nowait())
    #     except Empty:
    #         pass
    #     # self.false_boxes.put_nowait(None)
    #     # self.false_boxes.close()
    #     self.f_boxes = []
    #     try:
    #         while True:
    #             self.f_boxes.append(self.false_boxes.get_nowait())
    #     except Empty:
    #         pass

    # # self.unknown_boxes.put_nowait(None)
    # # # self.unknown_boxes.close()
    # # self.u_boxes = [b for b in iter(self.unknown_boxes.get, None)]
    # self.close()
    # self.closed = True

    def on_iteration(self):
        if self.internal_process_id == 1 and self.iteration.value % 10 == 0:
            self.plot(
                # self.true_boxes,
                # self.false_boxes,
                # self.unknown_boxes,
                title=f"Residual = {self.statistics.current_residual}, |boxes|={self.statistics.num_unknown}",
            )
            pass
        self.iteration.value = self.iteration.value + 1

    def add_unknown(self, box: Union[Box, List[Box]]):
        if isinstance(box, list):
            for b in box:
                if b.width() > self.config.tolerance:
                    self.unknown_boxes.put(b, timeout=self.config.queue_timeout)
                    self.statistics.num_unknown.value += 1
        else:
            if box.width() > self.config.tolerance:
                self.unknown_boxes.put(box)
                with self.statistics.num_unknown.get_lock():
                    self.statistics.num_unknown.value += 1

    def add_false(self, box: Box):
        # self.false_boxes.put(box)
        self.false_boxes.append(box)
        with self.statistics.num_false.get_lock():
            self.statistics.num_false.value += 1
        self.statistics.iteration_operation.put("f")

    def add_true(self, box: Box):
        # self.true_boxes.put(box)
        self.true_boxes.append(box)
        with self.statistics.num_true.get_lock():
            self.statistics.num_true.value += 1
        self.statistics.iteration_operation.put("t")

    def get_unknown(self):
        box = self.unknown_boxes.get(timeout=self.config.queue_timeout)
        self.statistics.num_unknown.value = self.statistics.num_unknown.value - 1
        self.statistics.current_residual.value = box.width()
        self.statistics.residuals.put(box.width())
        this_time = datetime.now()
        # FIXME self.statistics.iteration_time.put(this_time - self.statistics.last_time.value)
        # FIXME self.statistics.last_time[:] = str(this_time)
        return box

    def plotBox(self, interval: Interval = Interval(-20, 20)):
        box = Box(self.problem.parameters)
        for p, _ in box.bounds.items():
            box.bounds[p] = interval
        return box

    def plot(
        self,
        plot_bounds: Box = None,
        title="Feasible Regions",
    ):
        if not plot_bounds:
            plot_bounds = self.plotBox()

        clear_output(wait=True)

        if self.num_parameters == 1:
            self.plot1D(plot_bounds=plot_bounds)
        elif self.num_parameters == 2:
            self.plot2D(plot_bounds=plot_bounds)
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

    def plot1D(
        self,
        true_boxes: Queue = None,
        false_boxes: Queue = None,
        unknown_boxes: Queue = None,
        plot_bounds: Box = None,
    ):
        p1 = self.problem.parameters[0]

        true_boxes = true_boxes if true_boxes else self.true_boxes
        false_boxes = false_boxes if false_boxes else self.false_boxes
        unknown_boxes = unknown_boxes if unknown_boxes else self.unknown_boxes

        clear_output(wait=True)
        for i in iter(true_boxes.get, None):
            if plot_bounds.intersects(i) and i.finite():
                self.plot1DBox(i, p1, color="g")

        for i in iter(false_boxes.get, None):
            if plot_bounds.intersects(i) and i.finite():
                self.plot1DBox(i, p1, color="r")
        plt.xlabel(p1.name)
        plt.xlim([plot_bounds.bounds[p1].lb, plot_bounds.bounds[p1].ub])

    def _get_box_queue(self, q: Queue, q_size: int):
        result = []
        # q.put(None)
        try:
            # while q_size > 0 and not q.empty():
            while not q.empty():
                # q_size -= 1
                val = q.get_nowait()
                # if val:
                result.append(val)
                # else:
                # break
        except Empty:
            pass

        for b in result:
            q.put_nowait(b)

        return result

    def plot2D(
        self,
        true_boxes: Queue = None,
        false_boxes: Queue = None,
        unknown_boxes: Queue = None,
        plot_bounds: Box = None,
    ):
        p1 = self.problem.parameters[0]
        p2 = self.problem.parameters[1]

        true_boxes = true_boxes if true_boxes else self.true_boxes
        false_boxes = false_boxes if false_boxes else self.false_boxes
        unknown_boxes = unknown_boxes if unknown_boxes else self.unknown_boxes

        if self.closed:
            t_boxes = self.t_boxes
            f_boxes = self.f_boxes
        else:
            # with self.statistics.num_true.get_lock():
            #     n_true = self.statistics.num_true.value
            # t_boxes = self._get_box_queue(true_boxes, n_true)

            # with self.statistics.num_false.get_lock():
            #     n_false = self.statistics.num_false.value
            # f_boxes = self._get_box_queue(false_boxes, n_false)
            t_boxes = true_boxes
            f_boxes = false_boxes

        for b in t_boxes:
            if b and plot_bounds.intersects(b) and b.finite():
                self.plot2DBox(b, p1, p2, color="g")
        for b in f_boxes:
            if b and plot_bounds.intersects(b) and b.finite():
                self.plot2DBox(b, p1, p2, color="r")

        # tmp = Queue()
        # false_boxes.put(None)
        # j = 0
        # print("Begin Plotting.")
        # try:
        #     while not false_boxes.empty():
        #         # for i in iter(false_boxes.get_nowait, None):
        #         # print("Getting...")
        #         i = false_boxes.get(block=False, timeout=self.config.queue_timeout)
        #         # print(f"Got {i}")
        #         if i:
        #             # print("Putting...")
        #             tmp.put_nowait(i)
        #             # print(f"Put {i}")
        #         j += 1
        #         if plot_bounds.intersects(i) and i.finite():
        #             self.plot2DBox(i, p1, p2, color="r")
        # except Empty:
        #     pass
        # # print(f"Moved {j} boxes from false to tmp.")
        # # tmp.put_nowait(None)
        # j = 0
        # try:
        #     # for i in iter(tmp.get_nowait, None):
        #     while not tmp.empty():
        #         i = tmp.get_nowait()
        #         if i:
        #             false_boxes.put_nowait(i)
        #         j += 1
        # except Empty:
        #     pass
        # print(f"Moved {j} boxes from tmp to false.")
        # # # true_boxes.put(None)
        # j = 0
        # try:
        #     # for i in iter(true_boxes.get_nowait, None):
        #     while not true_boxes.empty():
        #         i = true_boxes.get_nowait()
        #         if i:
        #             tmp.put_nowait(i)
        #         j += 1
        #         if plot_bounds.intersects(i) and i.finite():
        #             self.plot2DBox(i, p1, p2, color="g")
        # except Empty:
        #     pass
        # print(f"Moved {j} boxes from true to tmp.")
        # # tmp.put(None)
        # j = 0
        # try:
        #     # for i in iter(tmp.get_nowait, None):
        #     while not tmp.empty():
        #         i = tmp.get_nowait()
        #         if i:
        #             true_boxes.put_nowait(i)
        #         j += 1
        # except Empty:
        #     pass
        # print(f"Moved {j} boxes from tmp to true.")

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
        episode.statistics.iteration_operation.put("s")

    def expand(self, rval: Queue, episode: BoxSearchEpisode):
        if episode.internal_process_id == 0:
            episode.add_unknown(episode.initial_box())

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
        episode.close()
        rval.put({"true_boxes": episode.true_boxes, "false_boxes": episode.false_boxes})
        # return True

    def search(
        self, problem: ParameterSynthesisScenario, config: SearchConfig = SearchConfig()
    ) -> Dict[str, List[Box]]:

        episode = BoxSearchEpisode(config, problem)
        self.episodes.append(episode)
        episode.on_start()

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
            episode.internal_process_id += 1

        results = [rval.get() for p in processes]
        rval.close()

        for p in processes:
            p.terminate()

        # completing process
        for p in processes:
            p.join()

        episode.false_boxes = [b for r in results for b in r["false_boxes"]]
        episode.true_boxes = [b for r in results for b in r["true_boxes"]]

        # episode.read_queues()
        # episode.close()  # Main process close queues

        # for p in processes:
        #     r = rval.get()
        #     print(f"Got {r} iterations from process")

        # true_boxes = []
        # try:
        #     while not episode.true_boxes.empty():
        #         true_boxes.append(episode.true_boxes.get_nowait())
        # except Empty:
        #     pass
        # false_boxes = []
        # num_false = episode.statistics.num_false.acquire(block=False)
        # try:
        #     while num_false > 0 and not episode.false_boxes.empty():
        #         num_false -= 1
        #         false_boxes.append(episode.false_boxes.get_nowait())
        # except Empty:
        #     pass

        return episode
