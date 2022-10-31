"""
This submodule defines the representations for episodes (single executions) of
one of the search algorithms supported by FUNMAN.
"""
from abc import ABC
from typing import List, Union
from datetime import datetime
from queue import Queue as SQueue

from funman.search_utils import Box, Point, SearchConfig, SearchStatistics

import multiprocessing as mp
from multiprocessing.managers import SyncManager

import logging
l = logging.getLogger(__file__)
l.setLevel(logging.INFO)

class SearchEpisode(ABC):
    def __init__(self, config: SearchConfig, problem, manager : SyncManager) -> None:
        self.config : SearchConfig = config
        self.problem  = problem
        self.num_parameters : int = len(self.problem.parameters)
        self.statistics = SearchStatistics(manager)


class BoxSearchEpisode(SearchEpisode):
    def __init__(self, config: SearchConfig, problem, manager : SyncManager) -> None:
        super(BoxSearchEpisode, self).__init__(config, problem, manager)
        self.unknown_boxes = manager.Queue()
        self.true_boxes = []
        self.false_boxes = []
        self.true_points = set({})
        self.false_points = set({})

        self.iteration = manager.Value("i", 0)


    def initialize_boxes(self, expander_count):
        self.add_unknown(self.initial_box())
        # initial_boxes = []
        # initial_boxes.append(self.initial_box())
        # num_boxes = 1
        # while num_boxes < 2 * (self.config.number_of_processes - 1):
        #     b1, b2 = initial_boxes.get().split()
        #     initial_boxes.put(b1)
        #     initial_boxes.put(b2)
        #     num_boxes += 1
        # for i in range(num_boxes):
        #     b = initial_boxes.get()
        #     self.add_unknown(b)
        #     l.debug(f"Initial box: {b}")

    def initial_box(self) -> Box:
        return Box(self.problem.parameters)

    def on_start(self):
        self.statistics.last_time.value = str(datetime.now())

    # def close(self):
    #     if self.multiprocessing:
    #         self.unknown_boxes.close()
    #         self.statistics.close()
    #         self.boxes_to_plot.close()

    def on_iteration(self):
        self.iteration.value = self.iteration.value + 1

    def _add_unknown_box(self, box: Box) -> bool:
        if box.width() > self.config.tolerance:
            self.unknown_boxes.put(box)
            self.statistics.num_unknown.value += 1
            return True
        return False

    def add_unknown(self, box: Union[Box, List[Box]]):
        did_add = False
        if isinstance(box, list):
            for b in box:
                did_add |= self._add_unknown_box(b)
        else:
            did_add = self._add_unknown_box(box)
        return did_add

    def add_false(self, box: Box):
        self.false_boxes.append(box)
        # with self.statistics.num_false.get_lock():
        #     self.statistics.num_false.value += 1
        # self.statistics.iteration_operation.put("f")

    def add_false_point(self, point: Point):
        if point in self.true_points:
            l.error(f"Point: {point} is marked false, but already marked true.")
        self.false_points.add(point)

    def add_true(self, box: Box):
        self.true_boxes.append(box)
        # with self.statistics.num_true.get_lock():
        #     self.statistics.num_true.value += 1
        # self.statistics.iteration_operation.put("t")

    def add_true_point(self, point: Point):
        if point in self.false_points:
            l.error(f"Point: {point} is marked true, but already marked false.")
        self.true_points.add(point)

    def get_unknown(self):
        box = self.unknown_boxes.get(timeout=self.config.queue_timeout)
        self.statistics.num_unknown.value = self.statistics.num_unknown.value - 1
        self.statistics.current_residual.value = box.width()
        self.statistics.residuals.put(box.width())
        this_time = datetime.now()
        # FIXME self.statistics.iteration_time.put(this_time - self.statistics.last_time.value)
        # FIXME self.statistics.last_time[:] = str(this_time)
        return box

    def get_box_to_plot(self):
        return self.boxes_to_plot.get(timeout=self.config.queue_timeout)

    def extract_point(self, model):
        point = Point(self.problem.parameters)
        for p in self.problem.parameters:
            point.values[p] = float(model[p.symbol].constant_value())
        return point
