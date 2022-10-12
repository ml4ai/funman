from typing import List, Union
from datetime import datetime
from multiprocessing import Queue, Value

from funman.search_utils import Box, Point, SearchConfig, SearchStatistics


class SearchEpisode(object):
    def __init__(self) -> None:
        self.statistics = SearchStatistics()


class BoxSearchEpisode(SearchEpisode):
    def __init__(self, config: SearchConfig, problem) -> None:
        super(BoxSearchEpisode, self).__init__()
        self.unknown_boxes = Queue()
        self.boxes_to_plot = Queue()
        self.true_boxes = []
        self.false_boxes = []

        self.config = config
        self.problem = problem
        self.num_parameters = len(self.problem.parameters)
        self.iteration = Value("i", 0)
        self.internal_process_id = 0

    def initial_box(self) -> Box:
        return Box(self.problem.parameters)

    def on_start(self):
        self.statistics.last_time.value = str(datetime.now())

    def close(self):
        self.unknown_boxes.close()
        self.statistics.close()
        self.boxes_to_plot.close()

    def on_iteration(self):
        self.iteration.value = self.iteration.value + 1

    def add_unknown(self, box: Union[Box, List[Box]]):
        if isinstance(box, list):
            for b in box:
                if b.width() > self.config.tolerance:
                    self.unknown_boxes.put(b, timeout=self.config.queue_timeout)
                    self.statistics.num_unknown.value += 1
                self.boxes_to_plot.put({"box": b, "label": "unknown"})
        else:
            if box.width() > self.config.tolerance:
                self.unknown_boxes.put(box)
                with self.statistics.num_unknown.get_lock():
                    self.statistics.num_unknown.value += 1
                self.boxes_to_plot.put({"box": box, "label": "unknown"})

    def add_false(self, box: Box):
        self.false_boxes.append(box)
        with self.statistics.num_false.get_lock():
            self.statistics.num_false.value += 1
        self.statistics.iteration_operation.put("f")
        self.boxes_to_plot.put({"box": box, "label": "false"})

    def add_false_point(self, point: Point):
        self.boxes_to_plot.put({"point": point, "label": "false"})

    def add_true(self, box: Box):
        self.true_boxes.append(box)
        with self.statistics.num_true.get_lock():
            self.statistics.num_true.value += 1
        self.statistics.iteration_operation.put("t")
        self.boxes_to_plot.put({"box": box, "label": "true"})

    def add_true_point(self, point: Point):
        self.boxes_to_plot.put({"point": point, "label": "true"})

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
        return self.boxes_to_plot.get()

    def extract_point(self, model):
        point = Point(self.problem.parameters)
        for p in self.problem.parameters:
            point.values[p] = float(model[p.symbol].constant_value())
        return point
